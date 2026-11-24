"""Rebuild math_dec_step.onnx as math_dec_step_kv.onnx: the same weights and the
same arithmetic, with per-layer cache inputs and outputs.

    python tools/math_dec_step_kv.py MODELS/math_dec_step.onnx MODELS/math_dec_step_kv.onnx

The original step graph takes each cache stacked over the
decoder layers ([layers, B, heads, P, d]) and indexes one layer per use.  ONNX
Runtime turns these indexes into a Split, which copies the constant cross-attention
keys and values on every token, and stacks the grown self-attention caches again
at the end of every step.

The rebuilt graph:
- takes self_k{i}, self_v{i}, cross_k{i}, cross_v{i} ([B, heads, P or T, d]) for
  each layer i in place of the stacked self_k, self_v, cross_k, cross_v,
- returns logits, new_k{i} (all layers), then new_v{i} (all layers),
- folds the shape and index nodes (Shape -> Gather on a static dimension, and the
  Unsqueeze, Concat, Slice ... after it) into initializers,
- keeps every node that computes an input of an arithmetic node at run time, as
  before.  The query slice of each cross-attention projection (weight and bias)
  stays a run-time Slice: as a constant, ONNX Runtime would prepack the weight
  and fuse the bias Add, and those kernels round differently.
- carries inferred shapes, so ONNX Runtime fuses the same nodes as before.

So the new graph gives the same logits and caches as the old one, bit for bit
(checked on random and real inputs, x86 and ARM, all optimization levels).
Needs the onnx package (not a runtime dependency of indra_ocr).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

CACHES = ("self_k", "self_v", "cross_k", "cross_v")
OUTPUTS = ("new_k", "new_v")
# Ops that only move, index or reshape data (or do integer shape arithmetic):
# folding them on constant inputs gives the exact values the runtime would.
MOVE_OPS = {
    "Constant", "Identity", "Shape", "Gather", "Unsqueeze", "Squeeze", "Concat",
    "Slice", "Reshape", "Transpose", "Cast", "Split",
}
INT_OPS = {"Add", "Sub", "Mul", "Div", "Equal", "Where", "ConstantOfShape"}


def convert(src: str, dst: str, fold: bool = True) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper, shape_inference
    from onnx.reference import ReferenceEvaluator

    m = onnx.load(src)
    g = m.graph
    ins = {i.name: i for i in g.input}
    for name in CACHES:
        if name not in ins:
            sys.exit(f"{src} has no input {name}: not a stacked-cache step graph")
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant" and n.attribute[0].name == "value":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

    rename: dict[str, str] = {}
    drop: set[int] = set()
    layers: set[int] = set()
    # 1. Gather(cache, i, axis=0) -> the new per-layer input.
    for k, n in enumerate(g.node):
        if n.op_type == "Gather" and n.input[0] in CACHES:
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            idx = consts.get(n.input[1])
            if axis != 0 or idx is None or idx.ndim != 0:
                sys.exit(f"unexpected use of {n.input[0]} in {n.name}")
            rename[n.output[0]] = f"{n.input[0]}{int(idx)}"
            layers.add(int(idx))
            drop.add(k)
    for n in g.node:
        if n.op_type != "Gather" and any(i in CACHES for i in n.input):
            sys.exit(f"{n.name} ({n.op_type}) reads a stacked cache directly")
    L = len(layers)
    if layers != set(range(L)):
        sys.exit(f"layers {sorted(layers)} are not 0..{L - 1}")

    # 2. new_k / new_v = Concat(Unsqueeze(k_i, 0)...) -> one output per layer.
    prod = {o: n for n in g.node for o in n.output}
    out_names: dict[str, list[str]] = {}
    for out in OUTPUTS:
        cat = prod[out]
        if cat.op_type != "Concat" or len(cat.input) != L:
            sys.exit(f"{out} is not a Concat of {L} layers")
        per = []
        for i, u in enumerate(cat.input):
            un = prod[u]
            ax = consts.get(un.input[1]) if len(un.input) > 1 else None
            if un.op_type != "Unsqueeze" or ax is None or ax.tolist() not in ([0], 0):
                sys.exit(f"{out} input {i} is not Unsqueeze(axis 0)")
            per.append(un.input[0])
        out_names[out] = per

    def r(x: str) -> str:
        return rename.get(x, x)

    nodes = [n for k, n in enumerate(g.node) if k not in drop]
    for n in nodes:
        n.input[:] = [r(x) for x in n.input]
    # Name each per-layer output tensor new_k{i}: rename where it is produced.
    out_rename = {}
    for out, per in out_names.items():
        for i, t in enumerate(per):
            out_rename[t] = f"{out}{i}"
    for n in nodes:
        n.input[:] = [out_rename.get(x, x) for x in n.input]
        n.output[:] = [out_rename.get(x, x) for x in n.output]

    def vi(name, like):
        t = like.type.tensor_type
        dims = [d.dim_param or d.dim_value for d in t.shape.dim][1:]
        return helper.make_tensor_value_info(name, t.elem_type, dims)

    new_inputs = []
    for i in g.input:
        if i.name in CACHES:
            new_inputs += [vi(f"{i.name}{k}", i) for k in range(L)]
        else:
            new_inputs.append(i)
    old_out = {o.name: o for o in g.output}
    new_outputs = [o for o in g.output if o.name not in OUTPUTS]
    for out in OUTPUTS:
        new_outputs += [vi(f"{out}{k}", old_out[out]) for k in range(L)]

    graph = helper.make_graph(nodes, g.name, new_inputs, new_outputs, list(g.initializer))
    m2 = helper.make_model(graph, opset_imports=m.opset_import, producer_name="indra math_dec_step_kv")
    m2.ir_version = m.ir_version

    if not fold:
        onnx.checker.check_model(m2)
        onnx.save(m2, dst)
        print(f"{src}: {len(g.node)} nodes -> {dst}: {len(nodes)} nodes, {L} layers (no folding)")
        return
    # 3. Shape -> Gather on a static dimension -> a constant.
    inf = shape_inference.infer_shapes(m2)
    shapes = {}
    for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
        shapes[v.name] = [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
    for n in nodes:
        if n.op_type == "Shape" and not n.attribute and n.input[0] in shapes:
            shp = shapes[n.input[0]]
            for c in nodes:
                if c.op_type == "Gather" and c.input[0] == n.output[0]:
                    idx = consts.get(c.input[1])
                    if idx is not None and idx.ndim == 0:
                        dim = shp[int(idx)]
                        if dim:
                            consts[c.output[0]] = np.array(dim, np.int64)
            if all(d for d in shp):
                consts[n.output[0]] = np.array(shp, np.int64)

    # 4. The value of every node whose inputs are all constants (data movement and
    # integer math only).
    for n in nodes:
        if all(o in consts for o in n.output):
            continue  # folded in step 3
        fold = n.op_type in MOVE_OPS or (
            n.op_type in INT_OPS
            and all(consts.get(x) is not None and consts[x].dtype.kind in "iub" for x in n.input if x)
        )
        if fold and n.op_type == "Cast":
            to = next(a.i for a in n.attribute if a.name == "to")
            fold = to in (TensorProto.INT64, TensorProto.INT32, TensorProto.BOOL) and all(
                consts.get(x) is not None and consts[x].dtype.kind in "iub" for x in n.input
            )
        if fold and all((x in consts) for x in n.input if x):
            sub = helper.make_graph(
                [n], "fold",
                [helper.make_tensor_value_info(x, helper.np_dtype_to_tensor_dtype(consts[x].dtype), consts[x].shape) for x in n.input if x],
                [helper.make_tensor_value_info(o, TensorProto.UNDEFINED, None) for o in n.output],
            )
            res = ReferenceEvaluator(helper.make_model(sub, opset_imports=m.opset_import)).run(
                None, {x: consts[x] for x in n.input if x}
            )
            for o, v in zip(n.output, res):
                consts[o] = np.asarray(v)

    # 5. Fold only what feeds data movement.  An arithmetic node (MatMul, Add, Mul,
    # Cast, Pow ...) that read a run-time value before still reads one: keep the
    # chain that computes each such input.  As constants, ONNX Runtime would fold
    # them further, prepack the cross-attention query weight and fuse the bias Add
    # and the attention scale, and the fused kernels round differently (logits off
    # by up to about 1e-5).  This keeps every arithmetic result bit for bit.
    prod = {o: n for n in nodes for o in n.output}
    source = {i.name for i in g.initializer}
    protect: set[str] = set()
    roots = {
        x
        for n in nodes
        if (n.op_type not in MOVE_OPS or n.op_type == "Cast") and not all(o in consts for o in n.output)
        for x in n.input
        if x in consts and x not in source and x in prod and prod[x].op_type != "Constant"
    }
    stack = list(roots)
    while stack:
        t = stack.pop()
        if t in protect or t not in prod or t not in consts:
            continue
        protect.add(t)
        stack += list(prod[t].input)
    keep = [n for n in nodes if any(o not in consts or o in protect for o in n.output)]
    kept_out = {o for n in keep for o in n.output}
    alias = {}
    for n in keep:
        mine = any(o in protect for o in n.output)
        for k, x in enumerate(n.input):
            if x in kept_out and x in consts and x not in roots and not mine:
                alias[x] = x + "__const"  # a node outside the chain reads the value
                n.input[k] = alias[x]
    for x, a in alias.items():
        consts[a] = consts[x]
    while True:  # drop nodes whose outputs nothing reads any more
        need = {x for n in keep for x in n.input} | {o.name for o in new_outputs}
        live = [n for n in keep if any(o in need for o in n.output)]
        if len(live) == len(keep):
            break
        keep = live
    kept_out = {o for n in keep for o in n.output}

    # 6. Constants the kept nodes read become initializers; drop everything unused.
    used = {x for n in keep for x in n.input} | {o.name for o in new_outputs}
    produced = kept_out | {i.name for i in new_inputs}
    inits = [numpy_helper.from_array(np.asarray(consts[x]), x) for x in sorted(used - produced) if x in consts]
    missing = used - produced - {i.name for i in inits} - {""}
    if missing:
        sys.exit(f"unresolved tensors: {sorted(missing)[:5]}")
    graph = helper.make_graph(keep, g.name, new_inputs, new_outputs, inits)
    m3 = helper.make_model(graph, opset_imports=m.opset_import, producer_name="indra math_dec_step_kv")
    m3.ir_version = m.ir_version
    m3 = shape_inference.infer_shapes(m3)
    onnx.checker.check_model(m3)
    onnx.save(m3, dst)
    print(f"{src}: {len(g.node)} nodes -> {dst}: {len(keep)} nodes, {L} layers")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--no-fold", action="store_true", help="only split the caches")
    a = ap.parse_args()
    convert(a.src, a.dst, fold=not a.no_fold)


if __name__ == "__main__":
    main()
