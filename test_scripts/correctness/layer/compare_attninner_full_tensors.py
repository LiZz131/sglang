#!/usr/bin/env python3
"""
Offline **strict numerical** comparison for attention inner **CPU full tensors**:

1. ``attninner_static_layer_*_w_kc_tp*.pt`` — one file per ``(layer, tp)`` (no req/pos in name).
2. ``attninner_layer_*_decode_*_req*.pt`` — decode checkpoints whose payload contains ``full_tensors``
   (requires ``SGLANG_DEBUG_DUMP_ATTN_INNER_SAVE_FULL=1`` and non-empty ``FULL_KEYS`` when dumping).

Req alignment follows the same idea as ``compare_decode_pipeline.py`` (layer index first, then
attninner-only mapping fallback). Per-slice paths use ``_find_attninner_path``-style lookup when
req ids differ between runs.

Usage::

python3 compare_attninner_full_tensors.py \
  --a run_0503_decode_special_dpattn_mlp_attn/normal_forward \
  --b run_0503_decode_dpattn_mlp_attn/normal_forward \
  -o run_0503_special_vs_dpattn.attninner.fulltensor.diff

Options::

  --chunk N       Elements per chunk when scanning flattened tensors (default 8_000_000).
  --only-keys     Comma subset of ``full_tensors`` keys to compare (default: all common keys).
  --skip-static   Do not compare ``attninner_static_*_w_kc*.pt``.
  --skip-decode   Do not compare decode ``attninner_layer_*`` payloads.
  --strict        Exit non-zero if any compared tensor has max_abs > atol or cos mismatch.
  --atol          max_abs threshold for --strict (default 0).
  --cos-tol       For --strict, fail if abs(1-cos) > this (default 1e-5).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, TextIO, Tuple

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import compare_decode_attn_inner as cai  # noqa: E402
import compare_decode_layer_hiddens as cdl  # noqa: E402
import compare_decode_mlp_inner as cmi  # noqa: E402
from compare_decode_pipeline import (  # noqa: E402
    _decode_body_for_pos,
    _find_attninner_path,
    _tp_map_for_layer_pair,
    _triples_for_req_on_a,
)


_STATIC_W_KC_RE = re.compile(
    r"^attninner_static_layer_(?P<layer>\d{4})_w_kc_tp(?P<tp>\d+)\.pt$"
)


class _TeeStdout:
    def __init__(self, a: TextIO, b: TextIO) -> None:
        self._a = a
        self._b = b

    def write(self, s: str) -> int:
        self._a.write(s)
        self._b.write(s)
        return len(s)

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()

    def isatty(self) -> bool:
        return self._a.isatty()


def _load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _iter_static_w_kc(dir_path: str) -> List[str]:
    out: List[str] = []
    try:
        names = os.listdir(dir_path)
    except FileNotFoundError:
        return []
    for fn in names:
        if fn.startswith("attninner_static_") and fn.endswith(".pt") and "_w_kc_" in fn:
            out.append(os.path.join(dir_path, fn))
    out.sort()
    return out


def _parse_static_w_kc(path: str) -> Optional[Tuple[int, int]]:
    m = _STATIC_W_KC_RE.match(os.path.basename(path))
    if not m:
        return None
    return int(m.group("layer")), int(m.group("tp"))


def _compare_flat_tensors_chunked(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    chunk: int,
) -> Dict[str, float]:
    if a.shape != b.shape:
        return {
            "shape_match": 0.0,
            "max_abs": float("inf"),
            "mean_abs": float("nan"),
            "cos": float("nan"),
            "numel": float(max(a.numel(), b.numel())),
        }
    a64 = a.reshape(-1).double()
    b64 = b.reshape(-1).double()
    n = int(a64.numel())
    if n == 0:
        return {"shape_match": 1.0, "max_abs": 0.0, "mean_abs": 0.0, "cos": 1.0, "numel": 0.0}

    max_abs = 0.0
    sum_abs = 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        xa, xb = a64[sl], b64[sl]
        max_abs = max(max_abs, float((xa - xb).abs().max().item()))
        sum_abs += float((xa - xb).abs().sum().item())
        dot += float((xa * xb).sum().item())
        na += float((xa * xa).sum().item())
        nb += float((xb * xb).sum().item())

    mean_abs = sum_abs / n
    cos = dot / (math.sqrt(na) * math.sqrt(nb) + 1e-30)
    return {
        "shape_match": 1.0,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cos": cos,
        "numel": float(n),
    }


def _strict_fail_tensor(m: Dict[str, float], *, atol: float, cos_tol: float) -> bool:
    if m.get("shape_match", 1.0) < 0.5:
        return True
    if m["max_abs"] > atol:
        return True
    cos = m["cos"]
    if not math.isfinite(cos):
        return True
    return abs(1.0 - cos) > cos_tol


def _compare_static_w_kc_dirs(
    dir_a: str,
    dir_b: str,
    *,
    chunk: int,
    strict: bool,
    atol: float,
    cos_tol: float,
) -> int:
    bad = 0
    paths_a = {p: _parse_static_w_kc(p) for p in _iter_static_w_kc(dir_a)}
    paths_b = {p: _parse_static_w_kc(p) for p in _iter_static_w_kc(dir_b)}
    keys_a = {v: k for k, v in paths_a.items() if v is not None}
    keys_b = {v: k for k, v in paths_b.items() if v is not None}
    common = sorted(set(keys_a) & set(keys_b))
    if not common:
        print("== static w_kc: no matching (layer,tp) pairs in both dirs ==")
        return 0

    print(f"== static w_kc: comparing {len(common)} (layer,tp) pair(s) ==")
    for lt in common:
        pa, pb = keys_a[lt], keys_b[lt]
        da, db = _load(pa), _load(pb)
        wa, wb = da.get("w_kc"), db.get("w_kc")
        if not torch.is_tensor(wa) or not torch.is_tensor(wb):
            print(f"--- static w_kc layer={lt[0]} tp={lt[1]}: missing tensor A={pa}")
            bad += 1
            continue
        m = _compare_flat_tensors_chunked(wa, wb, chunk=chunk)
        line = (
            f"static w_kc layer={lt[0]:04d} tp={lt[1]}: "
            f"max_abs={m['max_abs']:.10e} mean_abs={m['mean_abs']:.10e} "
            f"cos={m['cos']:.12f} numel={int(m['numel'])}"
        )
        print(line)
        if strict and _strict_fail_tensor(m, atol=atol, cos_tol=cos_tol):
            bad += 1
    return bad


def _compare_decode_full_tensors(
    dir_a: str,
    dir_b: str,
    *,
    chunk: int,
    only_keys: Optional[Set[str]],
    strict: bool,
    atol: float,
    cos_tol: float,
) -> int:
    bad = 0
    compared = 0

    req_layer_a = cdl._build_req_index(dir_a)
    req_layer_b = cdl._build_req_index(dir_b)
    mapping, notes = cdl._build_req_mapping(req_layer_a, req_layer_b)

    req_attn_a = cai._build_req_index(dir_a)
    req_attn_b = cai._build_req_index(dir_b)
    if not mapping:
        mapping2, notes2 = cai._build_req_mapping_attn(req_attn_a, req_attn_b)
        mapping.update(mapping2)
        notes.extend(notes2)

    req_inner_a = cmi._build_req_index(dir_a)
    req_inner_b = cmi._build_req_index(dir_b)

    for n in notes:
        print(f"# {n}")
    print(f"# mapped reqs: {len(mapping)}")

    for ra, rb in sorted(mapping.items()):
        ia = req_layer_a.get(ra, {})
        ib = req_layer_b.get(rb, {})
        atta = req_attn_a.get(ra, {})
        attb = req_attn_b.get(rb, {})

        tp_map = _tp_map_for_layer_pair(ia, ib)
        triples = _triples_for_req_on_a(
            ra,
            req_layer_a=req_layer_a,
            req_inner_a=req_inner_a,
            req_attn_a=req_attn_a,
        )
        if not triples:
            continue

        print(f"\n== req pair A={ra[:16]}… B={rb[:16]}… slices={len(triples)} ==")

        for pos, layer, tp_a in triples:
            tp_b = tp_map.get(tp_a, tp_a)
            stages = sorted(
                {
                    k.stage
                    for k in atta.keys()
                    if k.pos == pos and k.layer == layer and k.tp_rank == tp_a
                }
            )
            for stage in stages:
                pa = _find_attninner_path(
                    req_attn_a,
                    prefer_req=ra,
                    layer=layer,
                    stage=stage,
                    tp=tp_a,
                    pos=pos,
                )
                pb = _find_attninner_path(
                    req_attn_b,
                    prefer_req=rb,
                    layer=layer,
                    stage=stage,
                    tp=tp_b,
                    pos=pos,
                )
                if pa is None or pb is None:
                    continue
                da, db = _load(pa), _load(pb)
                fa = da.get("full_tensors") if isinstance(da.get("full_tensors"), dict) else {}
                fb = db.get("full_tensors") if isinstance(db.get("full_tensors"), dict) else {}
                if not fa and not fb:
                    continue
                keyset = set(fa) & set(fb)
                if only_keys is not None:
                    keyset &= only_keys
                if not keyset:
                    continue

                print(
                    f"\n>>> decode full_tensors  pos={pos} layer={layer} tp_A={tp_a} "
                    f"stage={stage}"
                )
                print(f"    A: {pa}")
                print(f"    B: {pb}")

                for k in sorted(keyset):
                    ta, tb = fa.get(k), fb.get(k)
                    if not torch.is_tensor(ta) or not torch.is_tensor(tb):
                        print(f"    [{k}] skip (not both tensors)")
                        bad += 1
                        continue
                    m = _compare_flat_tensors_chunked(ta, tb, chunk=chunk)
                    print(
                        f"    [{k}] max_abs={m['max_abs']:.10e} mean_abs={m['mean_abs']:.10e} "
                        f"cos={m['cos']:.12f} shape={tuple(ta.shape)} dtype_A={ta.dtype} dtype_B={tb.dtype}"
                    )
                    compared += 1
                    if strict and _strict_fail_tensor(m, atol=atol, cos_tol=cos_tol):
                        bad += 1

    print(f"\n--- summary decode full_tensors: tensor-pair comparisons={compared} bad_score={bad} ---")
    return bad


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--a", required=True, help="Run A normal_forward dir")
    p.add_argument("--b", required=True, help="Run B normal_forward dir")
    p.add_argument("-o", "--output", default="", help="Write report to file")
    p.add_argument("--chunk", type=int, default=8_000_000, help="Flat elements per chunk")
    p.add_argument(
        "--only-keys",
        default="",
        help="Comma-separated full_tensors keys (e.g. q_nope,q_nope_out,attn_output)",
    )
    p.add_argument("--skip-static", action="store_true", help="Skip attninner_static_*_w_kc*.pt")
    p.add_argument("--skip-decode", action="store_true", help="Skip decode attninner_layer_*")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--atol", type=float, default=0.0, help="max_abs fail threshold for --strict")
    p.add_argument("--cos-tol", type=float, default=1e-5)
    args = p.parse_args(list(argv) if argv is not None else None)

    dir_a = os.path.abspath(os.path.expanduser(args.a))
    dir_b = os.path.abspath(os.path.expanduser(args.b))
    if not os.path.isdir(dir_a) or not os.path.isdir(dir_b):
        print(f"ERROR: not a directory A={dir_a} B={dir_b}", file=sys.stderr)
        return 2

    only_keys: Optional[Set[str]] = None
    if (args.only_keys or "").strip():
        only_keys = {x.strip() for x in args.only_keys.split(",") if x.strip()}

    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else ""

    def run_body() -> int:
        total = 0
        if not args.skip_static and (only_keys is None or "w_kc" in only_keys):
            total += _compare_static_w_kc_dirs(
                dir_a,
                dir_b,
                chunk=args.chunk,
                strict=args.strict,
                atol=args.atol,
                cos_tol=args.cos_tol,
            )
        if not args.skip_decode:
            total += _compare_decode_full_tensors(
                dir_a,
                dir_b,
                chunk=args.chunk,
                only_keys=only_keys,
                strict=args.strict,
                atol=args.atol,
                cos_tol=args.cos_tol,
            )
        return 1 if args.strict and total > 0 else 0

    if out_path:
        with open(out_path, "w", encoding="utf-8") as out_f:
            ctx = (
                contextlib.redirect_stdout(_TeeStdout(sys.stdout, out_f))
                if sys.stdout.isatty()
                else contextlib.redirect_stdout(out_f)
            )
            with ctx:
                return run_body()
    return run_body()


if __name__ == "__main__":
    raise SystemExit(main())
