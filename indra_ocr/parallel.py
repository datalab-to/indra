"""Many pages on many cores: a pool of worker processes, one page per task.

ONNX Runtime's own threads speed up one page only a little: rendering, the text
layer and the layout steps run on one core.  Pages are independent, so one
single-threaded worker process per core uses the cores far better.  pdfium
is not thread-safe, so the workers are processes, not threads.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import native
from .pipeline import InputError, Page, is_pdf

THREADS_PER_WORKER = 1  # ORT threads in each worker process
MIN_POOL_PAGES = 4  # fewer pages: one process, all cores on each page
WORKER_MEMORY = 1.5e9  # bytes a worker can reach (giant scans); caps the pool

_conv = None
_doc: tuple = (None, None)


def usable_cores() -> int:
    """Cores this process may run on (taskset and cgroup CPU sets count)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def _max_workers_for_memory() -> int:
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # Windows
        return 1 << 30
    return max(1, int(ram // WORKER_MEMORY))


def _init(load_kwargs: dict) -> None:
    global _conv
    import cv2

    from . import load

    cv2.setNumThreads(load_kwargs["threads"])
    _conv = load(**load_kwargs)


def _run(unit: tuple):
    """One unit: (path, page index, password) of a PDF, or (path, None, max pages)
    of an image file.  Returns [(index, Page)], or the InputError."""
    global _doc
    path, index, extra = unit
    try:
        if index is None:
            return list(_conv.convert(path, extra))
        if _doc[0] != path:
            import pypdfium2 as pdfium

            _doc = (path, pdfium.PdfDocument(path, password=extra))
        doc = _doc[1]
        return [(index, _conv._safe(lambda: _conv.pdf_page(path, doc, index)))]
    except InputError as e:
        return e


def _plan(paths: list, max_pages: int, password: str | None) -> list:
    """Per input: its work units, or the InputError that stops it."""
    import pypdfium2 as pdfium

    plan = []
    for p in paths:
        path = str(p)
        if not Path(path).is_file():
            plan.append(InputError(f"{path}: no such file"))
        elif not is_pdf(path):
            plan.append([(path, None, max_pages)])
        else:
            try:
                doc = pdfium.PdfDocument(path, password=password)
                n = len(doc)
                doc.close()
            except pdfium.PdfiumError as e:
                locked = "password" in str(e).lower()
                plan.append(
                    InputError(
                        f"{path}: password-protected PDF (pass password=, or --password)"
                        if locked
                        else f"{path}: cannot open the PDF ({e})"
                    )
                )
                continue
            n = min(n, max_pages) if max_pages else n
            plan.append([(path, i, password) for i in range(n)])
    return plan


def convert_many(
    paths,
    cores: int | None = None,
    max_pages: int = 0,
    password: str | None = None,
    **load_kwargs,
):
    """Yield (path, result) per input, in input order: result is the list of
    (page index, Page) in page order, or the InputError that stopped the file.

    cores: how many cores to use (default: all this process may run on).  The
    pages of all inputs are spread over one worker process per core (fewer when
    physical memory is under WORKER_MEMORY per worker); with fewer than
    MIN_POOL_PAGES pages, one process uses all the cores on each page.
    load_kwargs go to indra_ocr.load (not threads)."""
    from .models import model_dir

    if max_pages < 0:
        raise ValueError(f"max_pages must be 0 (all pages) or more, not {max_pages}")
    # Resolve (and on a first run download) the models once, not in every worker.
    load_kwargs["path"] = str(
        model_dir(load_kwargs.get("path"), math=load_kwargs.get("math", False))
    )
    paths = [str(p) for p in paths]
    cores = max(1, cores or usable_cores())
    plan = _plan(paths, max_pages, password)
    n_pages = sum(len(u) for u in plan if isinstance(u, list))
    workers = min(cores // THREADS_PER_WORKER, n_pages, _max_workers_for_memory())
    if workers < 2 or n_pages < MIN_POOL_PAGES:
        yield from _serial(paths, plan, max_pages, password, cores, load_kwargs)
        return
    yield from _pooled(paths, plan, workers, max_pages, password, cores, load_kwargs)


def _serial(paths, plan, max_pages, password, threads: int, load_kwargs: dict):
    """The inputs in this process, ``threads`` ONNX Runtime threads on each page."""
    from . import load

    conv = load(threads=threads, **load_kwargs)
    for path, units in zip(paths, plan):
        if isinstance(units, InputError):
            yield path, units
            continue
        try:
            yield path, list(conv.convert(path, max_pages, password=password))
        except InputError as e:
            yield path, e


def _pooled(paths, plan, workers, max_pages, password, cores, load_kwargs: dict):
    """The pages on a pool of worker processes.  If the workers cannot run (with
    the spawn start method a worker imports the caller's main module, so on
    macOS and Windows a script must call convert_many under
    ``if __name__ == "__main__":``), warn and read the rest in this process."""
    import warnings
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from multiprocessing import get_context

    threads = THREADS_PER_WORKER
    flat = [u for units in plan if isinstance(units, list) for u in units]
    # Thread pools of the libraries the workers import (OpenMP, BLAS) take
    # their size from the environment at import: keep each to the worker's
    # share while the workers start (map submits every task, starting them all).
    env = {
        k: str(threads)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    done = 0  # inputs already yielded
    ex = ProcessPoolExecutor(
        workers,
        mp_context=get_context("spawn"),
        initializer=_init,
        initargs=({**load_kwargs, "threads": threads},),
    )
    try:
        try:
            results = ex.map(_run, flat)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for path, units in zip(paths, plan):
            if isinstance(units, InputError):
                yield path, units
                done += 1
                continue
            pages: list[tuple[int, Page]] = []
            error = None
            for _ in units:
                r = next(results)
                if isinstance(r, InputError):
                    error = error or r
                else:
                    pages += r
            yield path, (error if error is not None else pages)
            done += 1
    except BrokenProcessPool:
        warnings.warn(
            "indra_ocr.convert_many: the worker processes stopped (with the spawn "
            'start method, call convert_many under `if __name__ == "__main__":`); '
            "reading the remaining inputs in this process"
            + (", without the native kernels" if native.lib is not None else ""),
            RuntimeWarning,
            stacklevel=2,
        )
        # A worker that crashed may have crashed in a kernel: the rest runs as
        # Python code, so this process cannot crash the same way.
        with native.disabled():
            yield from _serial(
                paths[done:], plan[done:], max_pages, password, cores, load_kwargs
            )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
