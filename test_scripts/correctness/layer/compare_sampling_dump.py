#!/usr/bin/env python3
"""
Compare SGLANG sampling debug dumps (``dump_next_token_logits_topk`` payloads)
between a **tp** run and a **special_dp_attention** run under one base folder.

Layout (under ``--base``, e.g. ``run_add_logit/``)::

  tp/<req_id>/<phase>/pos_<n>/tp<r>_<stage>.pt
  special_dp_attention/...   (or ``special-dp-attention``)

Usage::

  python compare_sampling_dump.py --base run_add_logit -o run_add_logit.sampling.diff

``--base`` defaults to ``<this_dir>/run_add_logit``. Pairing key is the path relative to each mode root (``req/phase/pos_*/tp*_*.pt``).
Compared files are ordered **prefill → decode**, then by ``pos``, ``tp`` rank,
and stage (raw / pre_sampler / post_sampler), not raw lexicographic path order.

By default ``mode_tag`` mismatch is printed but **not** counted toward exit
status; use ``--strict-meta`` to count it. ``dp_rank`` is often expected to
differ between modes; use ``--strict-dp-rank`` to count ``dp_rank`` mismatches.
If the same relative path pairs two captures with different in-file ``req_id``
strings, use ``--ignore-payload-req-id`` so only logits / sampling tensors drive
the meta score (still printed as diff).

See also: ``compare_tensor.py`` for layer hidden compare (different payload).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple

import torch


class _TeeStdout:
    """Minimal text stream for ``contextlib.redirect_stdout``."""

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

# Payload keys from debug_sampling_dump.dump_next_token_logits_topk
META_SCALAR_KEYS = (
    "req_id",
    "phase",
    "pos",
    "tp_rank",
    "dp_rank",
    "tp_size",
    "mode_tag",
    "stage",
    "topk",
)
TENSOR_KEYS = ("topk_vals", "topk_idx")


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (list, tuple)):
        if len(v) > 16:
            return f"{type(v).__name__}(len={len(v)}, head={v[:8]!r} ...)"
        return repr(v)
    return repr(v)


def _load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def _tensor_stats(a: torch.Tensor, b: torch.Tensor, name: str) -> Tuple[str, Optional[Dict[str, float]]]:
    if a.shape != b.shape:
        return (
            f"    {name}: SHAPE_MISMATCH  a={tuple(a.shape)}  b={tuple(b.shape)}",
            None,
        )
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((af - bf) ** 2)).item())
    denom = float(torch.norm(af).item())
    rel = rmse / denom if denom > 1e-12 else float("nan")
    cos = float(
        (af.flatten() * bf.flatten()).sum().item()
        / (af.flatten().norm().item() * bf.flatten().norm().item() + 1e-12)
    )
    line = (
        f"    {name}: max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  "
        f"rmse={rmse:.6g}  rel_rmse={rel:.6g}  cos={cos:.8f}"
    )
    return line, {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "rel_rmse": rel,
        "cos": cos,
    }


def _integral_tensor_stats(
    a: torch.Tensor, b: torch.Tensor, name: str
) -> Tuple[str, Optional[Dict[str, float]]]:
    if a.shape != b.shape:
        return (
            f"    {name}: SHAPE_MISMATCH  a={tuple(a.shape)}  b={tuple(b.shape)}",
            None,
        )
    if a.dtype != b.dtype:
        return (
            f"    {name}: DTYPE_MISMATCH  a={a.dtype}  b={b.dtype}",
            None,
        )
    if torch.equal(a, b):
        return (
            f"    {name}: exact match  shape={tuple(a.shape)}  dtype={a.dtype}",
            {"max_abs": 0.0, "cos": 1.0},
        )
    d = (a.long() - b.long()).abs()
    max_abs = float(d.max().item())
    mism = int((d > 0).sum().item())
    line = (
        f"    {name}: int max_abs={max_abs:.6g}  mismatched_positions={mism}/"
        f"{a.numel()}  dtype={a.dtype}"
    )
    return line, {"max_abs": max_abs, "cos": 0.0}


def _strict_tensor_fail(m: Dict[str, float], cos_tol: float) -> bool:
    return m["max_abs"] > 0 and (
        not math.isfinite(m["cos"]) or abs(1.0 - m["cos"]) > cos_tol
    )


def _walk_pt_files(root: str) -> Dict[str, str]:
    """rel_path -> absolute path for all *.pt under root."""
    out: Dict[str, str] = {}
    root = os.path.abspath(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".pt"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            out[rel.replace(os.sep, "/")] = full
    return out


# Lexicographic sort puts ``decode`` before ``prefill``; use timeline order instead.
_STAGE_ORDER = {"raw": 0, "pre_sampler": 1, "post_sampler": 2}


def _sampling_dump_rel_sort_key(rel: str) -> Tuple:
    """Sort key: req folder, prefill then decode, pos, tp rank, stage, then rel."""
    parts = rel.replace(os.sep, "/").split("/")
    if len(parts) < 4:
        return ("", 9, 0, 0, 9, rel)
    req, phase, pos_dir, fname = parts[0], parts[1], parts[2], parts[3]
    phase_order = 0 if phase == "prefill" else 1 if phase == "decode" else 2
    pm = re.match(r"pos_(\d+)$", pos_dir)
    pos_i = int(pm.group(1)) if pm else 0
    fm = re.match(r"tp(\d+)_(raw|pre_sampler|post_sampler)\.pt$", fname)
    if fm:
        tp_r = int(fm.group(1))
        st_ord = _STAGE_ORDER.get(fm.group(2), 9)
    else:
        tp_r, st_ord = 0, 9
    return (req, phase_order, pos_i, tp_r, st_ord, rel)


def _normalize_special_name(name: str) -> str:
    return re.sub(r"[-_\s]+", "", name.lower())


def discover_mode_roots(base_dir: str) -> Tuple[str, str, List[str]]:
    """
    Return (tp_root, special_root, notes).

    * ``tp``: exact subdirectory ``tp``.
    * special: ``special_dp_attention``, ``special-dp-attention``, or a single
      child whose normalized name contains ``specialdpattention``.
    """
    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    notes: List[str] = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"not a directory: {base_dir}")

    tp_root = os.path.join(base_dir, "tp")
    if not os.path.isdir(tp_root):
        raise FileNotFoundError(f"missing ``tp`` under {base_dir}")

    candidates = (
        "special_dp_attention",
        "special-dp-attention",
        "special_dp-attention",
    )
    special_root: Optional[str] = None
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.isdir(p):
            special_root = p
            notes.append(f"special mode dir: {c}")
            break

    if special_root is None:
        subs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d != "tp"
        ]
        matches = [
            d
            for d in subs
            if "special" in d.lower()
            and "dp" in _normalize_special_name(d)
            and "attention" in d.lower()
        ]
        if len(matches) == 1:
            special_root = os.path.join(base_dir, matches[0])
            notes.append(f"special mode dir (heuristic): {matches[0]}")
        elif len(matches) > 1:
            # Prefer name closest to special_dp_attention
            matches.sort(
                key=lambda x: (
                    0 if "special_dp_attention" in x else 1,
                    len(x),
                    x,
                )
            )
            special_root = os.path.join(base_dir, matches[0])
            notes.append(
                f"multiple special-like dirs {matches}; using {matches[0]}"
            )

    if special_root is None:
        raise FileNotFoundError(
            f"could not find special_dp_attention (or variant) under {base_dir}"
        )

    return tp_root, special_root, notes


def compare_sampling_info(
    sa: Any,
    sb: Any,
    strict: bool,
    cos_tol: float,
    prefix: str = "    sampling_info",
) -> int:
    """Return bad count (0/1 increments for strict tensor failures)."""
    bad = 0
    if sa is None and sb is None:
        print(f"{prefix}: both None  [match]")
        return 0
    if sa is None or sb is None:
        print(
            f"{prefix}: one None  A={'set' if sa is not None else 'None'} "
            f"B={'set' if sb is not None else 'None'}  [diff]"
        )
        return 1
    if not isinstance(sa, dict) or not isinstance(sb, dict):
        print(f"{prefix}: not both dict  [diff]")
        return 1

    keys: Set[str] = set(sa.keys()) | set(sb.keys())
    for k in sorted(keys):
        va, vb = sa.get(k, None), sb.get(k, None)
        if va is None and vb is None:
            print(f"{prefix}.{k}: both None  [match]")
            continue
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape or va.dtype != vb.dtype:
                print(
                    f"{prefix}.{k}: shape/dtype  A={tuple(va.shape)} {va.dtype} "
                    f"B={tuple(vb.shape)} {vb.dtype}  [diff]"
                )
                bad += 1
                continue
            line, m = _tensor_stats(va, vb, f"{prefix}.{k}")
            print(line)
            if strict and m is not None and _strict_tensor_fail(m, cos_tol):
                bad += 1
            continue
        if torch.is_tensor(va) or torch.is_tensor(vb):
            print(
                f"{prefix}.{k}: tensor vs non-tensor  A={type(va)} B={type(vb)}  [diff]"
            )
            bad += 1
            continue
        ok = va == vb
        if isinstance(va, (list, tuple)) or isinstance(vb, (list, tuple)):
            ok = _list_equal(va, vb)
        tag = "[match]" if ok else "[diff]"
        print(f"{prefix}.{k}:  A={_fmt(va)}  B={_fmt(vb)}  {tag}")
        if not ok:
            bad += 1
    return bad


def print_and_compare_meta(
    da: Dict[str, Any],
    db: Dict[str, Any],
    *,
    strict_meta: bool,
    strict_dp_rank: bool,
    ignore_payload_req_id: bool,
) -> int:
    """Scalar / meta mismatch count (mode_tag excluded unless strict_meta)."""
    mismatches = 0
    print("  --- payload metainfo ---")
    for k in META_SCALAR_KEYS:
        va, vb = da.get(k), db.get(k)
        if k == "mode_tag" and not strict_meta:
            ok = va == vb
            tag = "[match]" if ok else "[diff, not counted; use --strict-meta]"
            print(f"  {k}:  A={_fmt(va)}  B={_fmt(vb)}  {tag}")
            if not ok:
                pass  # do not increment
            continue
        if k == "req_id" and ignore_payload_req_id:
            ok = va == vb
            tag = (
                "[match]"
                if ok
                else "[diff, not counted; use same client run or drop --ignore-payload-req-id]"
            )
            print(f"  {k}:  A={_fmt(va)}  B={_fmt(vb)}  {tag}")
            continue
        if k == "dp_rank" and not strict_dp_rank:
            ok = va == vb
            tag = "[match]" if ok else "[diff, not counted; use --strict-dp-rank]"
            print(f"  {k}:  A={_fmt(va)}  B={_fmt(vb)}  {tag}")
            if not ok:
                pass
            continue
        ok = va == vb
        if not ok:
            mismatches += 1
        tag = "[match]" if ok else "[diff]"
        print(f"  {k}:  A={_fmt(va)}  B={_fmt(vb)}  {tag}")

    extra_a = set(da.keys()) - set(
        META_SCALAR_KEYS + TENSOR_KEYS + ("sampling_info",)
    )
    extra_b = set(db.keys()) - set(
        META_SCALAR_KEYS + TENSOR_KEYS + ("sampling_info",)
    )
    if extra_a or extra_b:
        print(
            f"  extra_keys: only_in_A={sorted(extra_a)}  only_in_B={sorted(extra_b)}  [info]"
        )

    if mismatches:
        print(f"  metainfo: {mismatches} counted field(s) differ.")
    else:
        print("  metainfo: all counted fields match.")
    return mismatches


def compare_one_pair(
    rel: str,
    pa: str,
    pb: str,
    *,
    strict: bool,
    strict_meta: bool,
    strict_dp_rank: bool,
    ignore_payload_req_id: bool,
    cos_tol: float,
) -> int:
    """Return bad count for this file pair."""
    bad = 0
    print(f"=== {rel} ===")
    print(f"  A(tp): {pa}")
    print(f"  B(special): {pb}")
    try:
        da, db = _load(pa), _load(pb)
    except Exception as e:
        print(f"  LOAD_ERROR: {e}")
        return 1

    bad += print_and_compare_meta(
        da,
        db,
        strict_meta=strict_meta,
        strict_dp_rank=strict_dp_rank,
        ignore_payload_req_id=ignore_payload_req_id,
    )

    for k in TENSOR_KEYS:
        ta, tb = da.get(k), db.get(k)
        if ta is None and tb is None:
            print(f"  {k}: both None  [match]")
            continue
        if ta is None or tb is None:
            print(
                f"  {k}: one None  A={'set' if ta is not None else 'None'} "
                f"B={'set' if tb is not None else 'None'}  [diff]"
            )
            bad += 1
            continue
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            print(f"  {k}: not tensors  [diff]")
            bad += 1
            continue
        if k == "topk_idx" or (
            ta.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8)
            and tb.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8)
        ):
            line, m = _integral_tensor_stats(ta, tb, k)
        else:
            line, m = _tensor_stats(ta, tb, k)
        print(line)
        if strict and m is not None:
            if k == "topk_idx" or (
                ta.dtype in (torch.int32, torch.int64)
                and tb.dtype in (torch.int32, torch.int64)
            ):
                if m.get("max_abs", 0) > 0:
                    bad += 1
            elif _strict_tensor_fail(m, cos_tol):
                bad += 1

    bad += compare_sampling_info(
        da.get("sampling_info"),
        db.get("sampling_info"),
        strict=strict,
        cos_tol=cos_tol,
    )
    print()
    return bad


def main(argv: Optional[Iterable[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.join(here, "run_add_logit")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--base",
        default=default_base,
        help="Folder containing ``tp/`` and ``special_dp_attention/`` (default: sibling run_add_logit)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help="Write the same report to this file (UTF-8); stdout still prints if TTY",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on tensor cos / max_abs for topk tensors and sampling_info tensors",
    )
    p.add_argument("--cos-tol", type=float, default=1e-5, help="Cosine tolerance with --strict")
    p.add_argument(
        "--strict-meta",
        action="store_true",
        help="Count mode_tag mismatch in metainfo bad count",
    )
    p.add_argument(
        "--strict-dp-rank",
        action="store_true",
        help="Count dp_rank mismatch in metainfo bad count",
    )
    p.add_argument(
        "--ignore-payload-req-id",
        action="store_true",
        help="Do not count payload req_id mismatch (paths still must align); "
        "use when A/B runs used different client ids but same folder layout",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Compare at most N common files (0 = no limit), for quick smoke tests",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else ""

    def run_body() -> int:
        tp_root, sp_root, notes = discover_mode_roots(args.base)
        for n in notes:
            print(f"# {n}")
        print(f"base: {os.path.abspath(os.path.expanduser(args.base))}")
        print(f"tp:   {tp_root}")
        print(f"special: {sp_root}")
        print()

        files_tp = _walk_pt_files(tp_root)
        files_sp = _walk_pt_files(sp_root)
        common = sorted(
            set(files_tp.keys()) & set(files_sp.keys()),
            key=_sampling_dump_rel_sort_key,
        )
        only_tp = sorted(
            set(files_tp.keys()) - set(files_sp.keys()),
            key=_sampling_dump_rel_sort_key,
        )
        only_sp = sorted(
            set(files_sp.keys()) - set(files_tp.keys()),
            key=_sampling_dump_rel_sort_key,
        )

        print(f"tp .pt files: {len(files_tp)}")
        print(f"special .pt files: {len(files_sp)}")
        print(f"common relative paths: {len(common)}")
        print(
            "# compare order: prefill -> decode, then pos, tp rank, stage (raw / pre / post)"
        )
        if only_tp:
            head = only_tp[:20]
            more = f" ... (+{len(only_tp) - 20})" if len(only_tp) > 20 else ""
            print(f"only in tp ({len(only_tp)}): {head}{more}")
        if only_sp:
            head = only_sp[:20]
            more = f" ... (+{len(only_sp) - 20})" if len(only_sp) > 20 else ""
            print(f"only in special ({len(only_sp)}): {head}{more}")
        print()

        if not common:
            print("No common paths. Nothing to compare.")
            return 1

        if args.limit and args.limit > 0:
            common = common[: args.limit]
            print(f"(limited to first {args.limit} common files)")
            print()

        total_bad = 0
        for rel in common:
            total_bad += compare_one_pair(
                rel,
                files_tp[rel],
                files_sp[rel],
                strict=args.strict,
                strict_meta=args.strict_meta,
                strict_dp_rank=args.strict_dp_rank,
                ignore_payload_req_id=args.ignore_payload_req_id,
                cos_tol=args.cos_tol,
            )

        print("--- summary ---")
        print(f"compared: {len(common)} file pair(s)")
        print(f"bad score (meta + tensor strict): {total_bad}")
        if args.strict and total_bad > 0:
            print("strict: non-zero bad score -> exit 1", file=sys.stderr)
            return 1
        return 0

    if out_path:
        with open(out_path, "w", encoding="utf-8") as out_f:
            if sys.stdout.isatty():
                redirect_ctx = contextlib.redirect_stdout(_TeeStdout(sys.stdout, out_f))
            else:
                redirect_ctx = contextlib.redirect_stdout(out_f)
            with redirect_ctx:
                return run_body()
    return run_body()


if __name__ == "__main__":
    raise SystemExit(main())
