"""Cut a recognizer graph at its CTC head, for the native kernels.

A recognizer ends with its head conv (logits [B, V, T]) and then eight ops over
the transposed logits [B, T, V]: max, argmax, exp, sum and reciprocal give
``ids`` and ``conf``; log-softmax, top-k and a gather give the language-model
outputs ``blank``, ``topv`` and ``topi``.  With V = 11,750 classes (the CJK
recognizer) those ops take most of the recognizer's time, and ONNX Runtime runs
them whether or not their outputs are fetched.  The native kernels
(``ctc_greedy``, ``ctc_topk``) compute the same outputs from the logits in one
pass, so the graph is cut: its outputs become [logits, lengths] and the ops
after the head are dropped.

This reads and rewrites a few protobuf fields of the model (no onnx package):
ModelProto.graph; GraphProto node, initializer and output; NodeProto input,
output, op_type and attribute; ValueInfoProto name and type.
"""

from __future__ import annotations

from dataclasses import dataclass

# field numbers (onnx.proto)
MODEL_GRAPH = 7
GRAPH_NODE, GRAPH_INITIALIZER, GRAPH_OUTPUT = 1, 5, 12
NODE_INPUT, NODE_OUTPUT, NODE_OP_TYPE, NODE_ATTRIBUTE = 1, 2, 4, 5
ATTR_NAME, ATTR_INTS = 1, 8
TENSOR_NAME = 8
VALUE_NAME, VALUE_TYPE = 1, 2
TYPE_TENSOR, TENSOR_ELEM_TYPE, TENSOR_SHAPE = 1, 1, 2
SHAPE_DIM, DIM_VALUE = 1, 1
FLOAT = 1


@dataclass
class Field:
    num: int
    wire: int
    raw: bytes  # the whole field: key and value
    data: bytes = b""  # payload of a length-delimited field
    int_value: int = 0  # value of a varint field


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _fields(buf: bytes) -> list[Field]:
    out, i = [], 0
    while i < len(buf):
        start = i
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        data, value = b"", 0
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            i += 8
        elif wire == 2:
            n, i = _varint(buf, i)
            data, i = buf[i : i + n], i + n
        elif wire == 5:
            i += 4
        else:
            raise ValueError(f"protobuf wire type {wire} not supported")
        if i > len(buf):
            raise ValueError("truncated protobuf field")
        out.append(Field(num, wire, buf[start:i], data, value))
    return out


def _enc_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(num: int, payload: bytes) -> bytes:
    """A length-delimited field."""
    return _enc_varint(num << 3 | 2) + _enc_varint(len(payload)) + payload


def _strings(fields: list[Field], num: int) -> list[str]:
    return [f.data.decode() for f in fields if f.num == num and f.wire == 2]


def _ints(f: Field) -> list[int]:
    if f.wire == 0:
        return [f.int_value]
    vals, i = [], 0  # packed
    while i < len(f.data):
        v, i = _varint(f.data, i)
        vals.append(v)
    return vals


@dataclass
class _Node:
    field: Field
    op: str
    inputs: list[str]
    outputs: list[str]
    perm: list[int]


def _node(f: Field) -> _Node:
    fs = _fields(f.data)
    perm: list[int] = []
    for a in (x for x in fs if x.num == NODE_ATTRIBUTE):
        af = _fields(a.data)
        if _strings(af, ATTR_NAME) == ["perm"]:
            perm = [v for x in af if x.num == ATTR_INTS for v in _ints(x)]
    op = _strings(fs, NODE_OP_TYPE)
    return _Node(
        f,
        op[0] if op else "",
        _strings(fs, NODE_INPUT),
        _strings(fs, NODE_OUTPUT),
        perm,
    )


def _last_dim(value_info: bytes) -> int | None:
    """The last dimension of a tensor ValueInfoProto, when the graph fixes it."""
    for t in (f for f in _fields(value_info) if f.num == VALUE_TYPE):
        for tt in (f for f in _fields(t.data) if f.num == TYPE_TENSOR):
            for sh in (f for f in _fields(tt.data) if f.num == TENSOR_SHAPE):
                dims = [f for f in _fields(sh.data) if f.num == SHAPE_DIM]
                if dims:
                    v = [f for f in _fields(dims[-1].data) if f.num == DIM_VALUE]
                    return v[0].int_value if v else None
    return None


@dataclass
class Cut:
    model: bytes  # the cut graph: outputs [logits, lengths]
    outputs: list[str]  # the full graph's outputs
    topk: int  # topv's last dimension: the classes per step the LM path reads


def cut_ctc_head(model: bytes) -> Cut | None:
    """The recognizer graph ``model`` cut at its CTC head, or None when the
    graph does not end the way the recognizers do (then nothing is cut)."""
    try:
        top = _fields(model)
        graphs = [f for f in top if f.num == MODEL_GRAPH]
        if len(graphs) != 1:
            return None
        graph = _fields(graphs[0].data)
        nodes = [_node(f) for f in graph if f.num == GRAPH_NODE]
        outs = {
            _strings(_fields(f.data), VALUE_NAME)[0]: f
            for f in graph
            if f.num == GRAPH_OUTPUT
        }
        if not {"ids", "conf", "lengths"} <= set(outs):
            return None
        made_by = {o: n for n in nodes for o in n.outputs}
        argmax = made_by.get("ids")
        if argmax is None or argmax.op != "ArgMax" or not argmax.inputs:
            return None
        tr = made_by.get(argmax.inputs[0])
        if tr is None or tr.op != "Transpose" or tr.perm != [0, 2, 1]:
            return None
        logits = tr.inputs[0]
        # The nodes the new outputs need (graph nodes are in topological order).
        need = {logits, "lengths"}
        keep = set()
        for n in reversed(nodes):
            if any(o in need for o in n.outputs):
                keep.add(id(n.field))
                need.update(i for i in n.inputs if i)
        topk = _last_dim(outs["topv"].data) if "topv" in outs else None
        float_tensor = _enc_varint(TENSOR_ELEM_TYPE << 3) + _enc_varint(FLOAT)
        new_outs = (
            _ld(
                GRAPH_OUTPUT,
                _ld(VALUE_NAME, logits.encode())
                + _ld(VALUE_TYPE, _ld(TYPE_TENSOR, float_tensor)),
            )
            + outs["lengths"].raw
        )
        parts, placed = [], False
        for f in graph:
            if f.num == GRAPH_NODE and id(f) not in keep:
                continue
            if (
                f.num == GRAPH_INITIALIZER
                and not set(_strings(_fields(f.data), TENSOR_NAME)) & need
            ):
                continue
            if f.num == GRAPH_OUTPUT:
                if not placed:
                    parts.append(new_outs)
                    placed = True
                continue
            parts.append(f.raw)
        new_graph = b"".join(parts)
        body = b"".join(
            _ld(MODEL_GRAPH, new_graph) if f.num == MODEL_GRAPH else f.raw for f in top
        )
        return Cut(body, list(outs), topk or 8)
    except (IndexError, ValueError, UnicodeDecodeError):
        return None
