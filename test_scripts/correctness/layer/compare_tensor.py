#!/usr/bin/env python3
"""
Compare per-layer debug dumps from SGLANG_DEBUG_SAVE_LAYER_HIDDENS (normal_forward vs split_prefill).

For **decode** runs (two ``normal_forward/`` dirs, req-id mapping, ``layer_*`` + ``mlpinner_*`` in
pipeline order), prefer ``compare_decode_pipeline.py`` in this directory instead of this script.

Usage:
 python compare_tensor.py \
 --a run_add_logit/normal_forward \
 --b run_add_logit/split_prefill \
    --half > run_add_logit.diff

Or with defaults under this script's sibling ``run_1/`` layout.

Metainfo (version, layer_id, split_interval, ranks, forward_mode, globals,
``input_ids`` / ``positions`` / ``seq_lens*``) is printed and compared before
``hidden_states`` / ``residual``. ``path_tag`` is shown but not counted as a diff.
``--strict`` counts metainfo mismatches; use ``--no-meta-strict`` to only enforce
tensors.

``--half``: when ``B.shape[0] == 2 * A.shape[0]`` and other dims match, split ``B`` on
``dim=0`` into two slices and compare each to ``A`` (metainfo tensors + ``hidden_states`` /
``residual``). Strict passes if at least one half-pair is within ``--cos-tol`` (or exact).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

# Payload keys written by deepseek_v2._debug_save_layer_hidden_if_enabled
META_SCALAR_KEYS = (
    "version",
    "path_tag",
    "layer_id",
    "split_interval",
    "tp_rank",
    "attn_dp_rank",
    "forward_mode",
)
META_LIST_KEYS = ("global_num_tokens", "global_seq_lens_sum_per_dp")
META_TENSOR_KEYS = ("input_ids", "positions", "seq_lens_cpu", "seq_lens")


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (list, tuple)):
        if len(v) > 16:
            return f"{type(v).__name__}(len={len(v)}, head={v[:8]!r} ...)"
        return repr(v)
    return repr(v)


def _half_split_dim0(
    a: torch.Tensor, b: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """If B on dim0 is 2x A and trailing shapes match, return (B[0:na], B[na:2*na])."""
    if a.dim() < 1 or b.dim() < 1:
        return None
    if a.shape[0] * 2 != b.shape[0]:
        return None
    if a.shape[1:] != b.shape[1:]:
        return None
    n = a.shape[0]
    return b[:n], b[n : 2 * n]


def _strict_tensor_fail(m: Dict[str, float], cos_tol: float) -> bool:
    return m["max_abs"] > 0 and (
        not math.isfinite(m["cos"]) or abs(1.0 - m["cos"]) > cos_tol
    )


def _list_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        la, lb = list(a), list(b)
    except TypeError:
        return False
    if len(la) != len(lb):
        return False
    return all(x == y for x, y in zip(la, lb))


def _tensor_meta_line(
    key: str, ta: Optional[torch.Tensor], tb: Optional[torch.Tensor]
) -> Tuple[str, bool]:
    """Return (line, match). match False if mismatch (or incomparable)."""
    if ta is None and tb is None:
        return f"  {key}: both None  [match]", True
    if ta is None or tb is None:
        return (
            f"  {key}: one None  A={'set' if ta is not None else 'None'} "
            f"B={'set' if tb is not None else 'None'}  [diff]",
            False,
        )
    if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
        return f"  {key}: not tensors  [diff]", False
    sa, sb = tuple(ta.shape), tuple(tb.shape)
    dta, dtb = ta.dtype, tb.dtype
    head = f"  {key}: shape A={sa} B={sb}  dtype {dta} vs {dtb}"
    if sa != sb or dta != dtb:
        return f"{head}  [diff]", False
    if ta.numel() == 0 and tb.numel() == 0:
        return f"{head}  (empty)  [match]", True
    if torch.equal(ta, tb):
        return f"{head}  values [match]  [match]", True
    # same shape: numeric diff
    taf, tbf = ta.float(), tb.float()
    diff = (taf - tbf).abs()
    max_abs = float(diff.max().item())
    return (
        f"{head}  max_abs={max_abs:.6g}  (values differ)  [diff]",
        max_abs == 0.0,
    )


def _meta_tensor_mismatch(
    key: str,
    ta: Optional[torch.Tensor],
    tb: Optional[torch.Tensor],
    use_half: bool,
) -> int:
    """Print meta tensor row(s); return 1 if counted as mismatch, else 0."""
    if ta is None and tb is None:
        print(f"  {key}: both None  [match]")
        return 0
    if ta is None or tb is None:
        print(
            f"  {key}: one None  A={'set' if ta is not None else 'None'} "
            f"B={'set' if tb is not None else 'None'}  [diff]"
        )
        return 1
    if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
        print(f"  {key}: not tensors  [diff]")
        return 1
    if use_half:
        parts = _half_split_dim0(ta, tb)
        if parts is not None:
            b0, b1 = parts
            n0 = int(ta.shape[0])
            n1 = 2 * n0
            print(
                f"  {key}:  --half--  B.shape[0]==2*A.shape[0]  "
                f"-> compare A to B[0:{n0}] and A to B[{n0}:{n1}]"
            )
            s1, ok1 = _tensor_meta_line(f"{key}@B[0:{n0}]", ta, b0)
            s2, ok2 = _tensor_meta_line(f"{key}@B[{n0}:{n1}]", ta, b1)
            print(s1)
            print(s2)
            return 0 if (ok1 or ok2) else 1
    line, ok = _tensor_meta_line(key, ta, tb)
    print(line)
    return 0 if ok else 1


def print_and_compare_meta(
    da: Dict[str, Any], db: Dict[str, Any], use_half: bool = False
) -> int:
    """Print side-by-side metainfo; return mismatch count (``path_tag`` is never counted)."""
    print("  --- metainfo ---")
    mismatches = 0

    for k in META_SCALAR_KEYS:
        va, vb = da.get(k, None), db.get(k, None)
        if k == "path_tag":
            print(
                f"  {k}:  A={_fmt(va)}  B={_fmt(vb)}  "
                f"(not counted; expected to differ A vs B)"
            )
            continue
        ok = va == vb
        if not ok:
            mismatches += 1
        tag = "[match]" if ok else "[diff]"
        print(f"  {k}:  A={_fmt(va)}  B={_fmt(vb)}  {tag}")

    for k in META_LIST_KEYS:
        va, vb = da.get(k, None), db.get(k, None)
        if va is None and vb is None:
            print(f"  {k}:  both None  [match]")
            continue
        ok = _list_equal(va, vb)
        if not ok:
            mismatches += 1
        tag = "[match]" if ok else "[diff]"
        print(f"  {k}:  A={_fmt(va)}  B={_fmt(vb)}  {tag}")

    for k in META_TENSOR_KEYS:
        ta, tb = da.get(k), db.get(k)
        mismatches += _meta_tensor_mismatch(k, ta, tb, use_half)

    # any extra keys in one dict only
    keys_a, keys_b = set(da.keys()), set(db.keys())
    all_known = set(
        META_SCALAR_KEYS
        + META_LIST_KEYS
        + META_TENSOR_KEYS
        + ("hidden_states", "residual")
    )
    extra_a = keys_a - all_known
    extra_b = keys_b - all_known
    if extra_a or extra_b:
        print(f"  extra_keys: only_in_A={sorted(extra_a)}  only_in_B={sorted(extra_b)}  [info]")

    if mismatches:
        print(f"  metainfo: {mismatches} field(s) differ (lists/tensors/scalars).")
    else:
        print("  metainfo: all compared fields match (path_tag not counted).")
    return mismatches


def _tensor_stats(
    a: torch.Tensor, b: torch.Tensor, name: str
) -> Tuple[str, Optional[Dict[str, float]]]:
    if a.shape != b.shape:
        return (
            f"  {name}: SHAPE_MISMATCH  a={tuple(a.shape)}  b={tuple(b.shape)}",
            None,
        )
    if a.dtype != b.dtype:
        a = a.float()
        b = b.float()
    else:
        a = a.float()
        b = b.float()
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((a - b) ** 2)).item())
    denom = float(torch.norm(a).item())
    rel = rmse / denom if denom > 1e-12 else float("nan")
    # cosine similarity
    af = a.flatten()
    bf = b.flatten()
    cos = float(
        (af * bf).sum().item()
        / (af.norm().item() * bf.norm().item() + 1e-12)
    )
    lines = (
        f"  {name}: max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  "
        f"rmse={rmse:.6g}  rel_rmse={rel:.6g}  cos={cos:.8f}"
    )
    metrics = {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "rel_rmse": rel,
        "cos": cos,
    }
    return lines, metrics


def _compare_hidden_or_residual(
    ta: torch.Tensor,
    tb: torch.Tensor,
    key: str,
    use_half: bool,
    strict: bool,
    cos_tol: float,
) -> int:
    """Return contribution to `bad` under ``strict`` (0 or 1). Non-strict: 0 unless error."""
    if use_half:
        parts = _half_split_dim0(ta, tb)
        if parts is not None:
            b0, b1 = parts
            n0 = int(ta.shape[0])
            n1 = 2 * n0
            print(
                f"  {key}:  --half--  B.shape[0]==2*A.shape[0]  "
                f"-> compare A to B[0:{n0}] and A to B[{n0}:{n1}]"
            )
            l1, m1 = _tensor_stats(ta, b0, f"{key}  [A vs B[0:{n0}]]")
            l2, m2 = _tensor_stats(ta, b1, f"{key}  [A vs B[{n0}:{n1}]]")
            print(l1)
            print(l2)
            if m1 is None or m2 is None:
                return 1
            if not strict:
                return 0
            f1 = _strict_tensor_fail(m1, cos_tol)
            f2 = _strict_tensor_fail(m2, cos_tol)
            return 1 if (f1 and f2) else 0
    line, m = _tensor_stats(ta, tb, key)
    print(line)
    if m is None:
        return 1
    if not strict:
        return 0
    return 1 if _strict_tensor_fail(m, cos_tol) else 0


def _load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_a = os.path.join(here, "run_1", "normal_forward")
    default_b = os.path.join(here, "run_1", "split_prefill")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", default=default_a, help="Directory A (e.g. normal_forward)")
    p.add_argument("--b", default=default_b, help="Directory B (e.g. split_prefill)")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero if any tensor pair mismatches shape or cos < 1 - tol",
    )
    p.add_argument("--cos-tol", type=float, default=1e-5, help="Used with --strict for cos")
    p.add_argument(
        "--no-meta-strict",
        action="store_true",
        help="With --strict, do not fail on metainfo / input_ids / positions mismatches (only hidden tensors)",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="If B.shape[0]==2*A.shape[0] (other dims match), split B on dim0 and "
        "compare A to each half (meta tensors + hidden_states/residual). "
        "Strict: pass if at least one half is within --cos-tol (or max_abs=0).",
    )
    args = p.parse_args()

    dir_a = os.path.expanduser(args.a)
    dir_b = os.path.expanduser(args.b)

    if not os.path.isdir(dir_a):
        print(f"ERROR: not a directory: {dir_a}", file=sys.stderr)
        return 2
    if not os.path.isdir(dir_b):
        print(f"ERROR: not a directory: {dir_b}", file=sys.stderr)
        return 2

    files_a = {f for f in os.listdir(dir_a) if f.endswith(".pt")}
    files_b = {f for f in os.listdir(dir_b) if f.endswith(".pt")}
    common = sorted(files_a & files_b)
    only_a = sorted(files_a - files_b)
    only_b = sorted(files_b - files_a)

    print(f"A: {dir_a}  ({len(files_a)} .pt files)")
    print(f"B: {dir_b}  ({len(files_b)} .pt files)")
    if only_a:
        print(f"Only in A: {only_a[:10]}{' ...' if len(only_a) > 10 else ''}")
    if only_b:
        print(f"Only in B: {only_b[:10]}{' ...' if len(only_b) > 10 else ''}")
    if not common:
        print("No common .pt filenames between A and B. Nothing to compare.")
        return 1

    print(f"Comparing {len(common)} common file(s).")
    if args.half:
        print("  ( --half: B split on dim0 when 2*A[dim0] == B[dim0] )")
    print()

    bad = 0
    for name in common:
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        print(f"=== {name} ===")
        try:
            da, db = _load(pa), _load(pb)
        except Exception as e:
            print(f"  LOAD_ERROR: {e}")
            bad += 1
            continue

        meta_bad = print_and_compare_meta(da, db, use_half=args.half)
        if not args.no_meta_strict:
            bad += meta_bad

        for key in ("hidden_states", "residual"):
            ta, tb = da.get(key), db.get(key)
            if ta is None and tb is None:
                print(f"  {key}: both None (ok)")
                continue
            if ta is None or tb is None:
                print(f"  {key}: one is None -> A={ta is not None} B={tb is not None}")
                bad += 1
                continue
            if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
                print(f"  {key}: not tensors")
                bad += 1
                continue
            bad += _compare_hidden_or_residual(
                ta, tb, key, args.half, args.strict, args.cos_tol
            )
        print()

    if args.strict and bad > 0:
        print(f"strict: {bad} issue(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
