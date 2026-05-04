#!/usr/bin/env python3
"""
Compare sampling debug dumps (``dump_next_token_logits_topk`` payloads) across **two runs**
stored in different base folders, where request ids differ between servers.

Run layouts:
  - TP run:        <tp_run>/tp/<req_id>/<phase>/pos_<n>/tp<r>_<stage>.pt
  - DP-attn run:   <dp_run>/(special_dp_attention|dp_attention)/<req_id>/...
                  (``special_dp_*`` name variants accepted; standard dp-attn uses ``dp_attention/``)

This script auto-maps req ids across runs by a signature derived from file-tree stats:
  signature(req) = (prefill_max_pos, decode_min_pos, decode_max_pos, decode_pos_count)

This tends to be stable when the two runs are fully comparable (same prompts / lengths).
If collisions occur, the script falls back to a deterministic order pairing and reports it.

Usage:
  python compare_sampling_dump_two_runs.py \
    --tp-run /.../run_0430_decode_tp \
    --dp-run /.../run_0430_decode_dpattn \
    -o run_0430_decode.sampling.diff
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import torch


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


def _mode_dir_under(base: str, prefer: List[str]) -> str:
    for name in prefer:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    # heuristic fallback
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    for d in subs:
        if "special" in d.lower() and "attention" in d.lower():
            return os.path.join(base, d)
    raise FileNotFoundError(f"cannot find mode dir under {base}: tried {prefer}")


_POS_RE = re.compile(r"^pos_(\d+)$")


@dataclass(frozen=True)
class ReqSig:
    prefill_max_pos: int
    decode_min_pos: int
    decode_max_pos: int
    decode_pos_count: int


def _collect_req_sig(req_root: str) -> Optional[ReqSig]:
    prefill_dir = os.path.join(req_root, "prefill")
    decode_dir = os.path.join(req_root, "decode")

    prefill_max = -1
    if os.path.isdir(prefill_dir):
        for d in os.listdir(prefill_dir):
            m = _POS_RE.match(d)
            if m:
                prefill_max = max(prefill_max, int(m.group(1)))

    decode_positions: List[int] = []
    if os.path.isdir(decode_dir):
        for d in os.listdir(decode_dir):
            m = _POS_RE.match(d)
            if m:
                decode_positions.append(int(m.group(1)))

    if prefill_max < 0 or not decode_positions:
        return None
    decode_positions.sort()
    return ReqSig(
        prefill_max_pos=prefill_max,
        decode_min_pos=decode_positions[0],
        decode_max_pos=decode_positions[-1],
        decode_pos_count=len(decode_positions),
    )


def _list_reqs(mode_root: str) -> List[str]:
    reqs = []
    for d in os.listdir(mode_root):
        p = os.path.join(mode_root, d)
        if os.path.isdir(p):
            reqs.append(d)
    reqs.sort()
    return reqs


def _build_mapping(tp_root: str, dp_root: str) -> Tuple[Dict[str, str], List[str]]:
    notes: List[str] = []
    tp_reqs = _list_reqs(tp_root)
    dp_reqs = _list_reqs(dp_root)

    sig_to_tp: Dict[ReqSig, List[str]] = {}
    sig_to_dp: Dict[ReqSig, List[str]] = {}

    for rid in tp_reqs:
        sig = _collect_req_sig(os.path.join(tp_root, rid))
        if sig is not None:
            sig_to_tp.setdefault(sig, []).append(rid)
    for rid in dp_reqs:
        sig = _collect_req_sig(os.path.join(dp_root, rid))
        if sig is not None:
            sig_to_dp.setdefault(sig, []).append(rid)

    mapping: Dict[str, str] = {}
    common_sigs = sorted(set(sig_to_tp) & set(sig_to_dp), key=lambda s: (s.prefill_max_pos, s.decode_max_pos, s.decode_pos_count))
    for sig in common_sigs:
        tps = sorted(sig_to_tp[sig])
        dps = sorted(sig_to_dp[sig])
        n = min(len(tps), len(dps))
        if len(tps) != 1 or len(dps) != 1:
            notes.append(f"signature collision {sig}: tp={tps} dp={dps} -> pairing first {n} by sorted order")
        for i in range(n):
            mapping[tps[i]] = dps[i]

    # Fallback: if we couldn't map some reqs (e.g. missing decode), pair remaining by order.
    unm_tp = [r for r in tp_reqs if r not in mapping]
    used_dp = set(mapping.values())
    unm_dp = [r for r in dp_reqs if r not in used_dp]
    if unm_tp and unm_dp:
        notes.append(f"fallback order pairing: {len(unm_tp)} tp req(s) with {len(unm_dp)} dp req(s)")
        for a, b in zip(unm_tp, unm_dp):
            mapping[a] = b
    return mapping, notes


def _walk_pt_files(root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".pt"):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                out[rel] = full
    return out


def _tensor_equal_or_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, str]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype A={tuple(a.shape)} {a.dtype} B={tuple(b.shape)} {b.dtype}"
    if torch.equal(a, b):
        return True, "exact"
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    max_abs = float(diff.max().item())
    return False, f"max_abs={max_abs:.6g}"


def compare_payload(pa: str, pb: str) -> List[str]:
    da, db = _load(pa), _load(pb)
    lines: List[str] = []
    # Scalars (ignore req_id and mode_tag)
    for k in ("phase", "pos", "tp_rank", "stage", "topk", "tp_size"):
        if da.get(k) != db.get(k):
            lines.append(f"  meta.{k}: A={da.get(k)!r} B={db.get(k)!r} [diff]")
    # Tensors
    for k in ("topk_vals", "topk_idx"):
        ta, tb = da.get(k), db.get(k)
        if not torch.is_tensor(ta) or not torch.is_tensor(tb):
            lines.append(f"  {k}: missing/non-tensor [diff]")
            continue
        ok, info = _tensor_equal_or_stats(ta, tb)
        tag = "match" if ok else "diff"
        lines.append(f"  {k}: {info} [{tag}]")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tp-run", required=True, help="Base dir containing tp/ (TP server run)")
    p.add_argument(
        "--dp-run",
        required=True,
        help="Base dir containing special_dp_attention/ or dp_attention/ (decode sampling tree)",
    )
    p.add_argument("-o", "--output", default="", help="Write report to file (and tee to stdout if TTY)")
    p.add_argument("--limit", type=int, default=0, help="Compare at most N paired files (0 = no limit)")
    args = p.parse_args(list(argv) if argv is not None else None)

    tp_base = os.path.abspath(os.path.expanduser(args.tp_run))
    dp_base = os.path.abspath(os.path.expanduser(args.dp_run))

    tp_root = _mode_dir_under(tp_base, ["tp"])
    dp_root = _mode_dir_under(
        dp_base,
        [
            "special_dp_attention",
            "special-dp-attention",
            "special_dp-attention",
            "dp_attention",
        ],
    )

    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else ""

    def run_body() -> int:
        mapping, notes = _build_mapping(tp_root, dp_root)
        print(f"tp_root: {tp_root}")
        print(f"dp_root: {dp_root}")
        for n in notes:
            print(f"# {n}")
        print(f"mapped reqs: {len(mapping)}")
        if mapping:
            head = list(mapping.items())[:10]
            print(f"mapping head: {head}")
        print()

        bad = 0
        compared = 0

        # For each mapped req pair, compare common relative paths under that req.
        for tp_rid, dp_rid in sorted(mapping.items()):
            tp_req_root = os.path.join(tp_root, tp_rid)
            dp_req_root = os.path.join(dp_root, dp_rid)
            files_tp = _walk_pt_files(tp_req_root)
            files_dp = _walk_pt_files(dp_req_root)
            common = sorted(set(files_tp) & set(files_dp))
            if not common:
                print(f"== req pair tp={tp_rid} dp={dp_rid}: no common files ==")
                bad += 1
                continue
            print(f"== req pair tp={tp_rid} dp={dp_rid}: common_files={len(common)} ==")
            for rel in common:
                print(f"=== {tp_rid} :: {rel} ===")
                pa, pb = files_tp[rel], files_dp[rel]
                print(f"A: {pa}")
                print(f"B: {pb}")
                try:
                    for l in compare_payload(pa, pb):
                        print(l)
                except Exception as e:
                    print(f"  COMPARE_ERROR: {e}")
                    bad += 1
                print()
                compared += 1
                if args.limit and compared >= args.limit:
                    print("(limit reached)")
                    print("--- summary ---")
                    print(f"compared files: {compared}")
                    print(f"bad score: {bad}")
                    return 0 if bad == 0 else 1

        print("--- summary ---")
        print(f"compared files: {compared}")
        print(f"bad score: {bad}")
        return 0 if bad == 0 else 1

    if out_path:
        with open(out_path, "w", encoding="utf-8") as out_f:
            if sys.stdout.isatty():
                ctx: Any = contextlib.redirect_stdout(_TeeStdout(sys.stdout, out_f))
            else:
                ctx = contextlib.redirect_stdout(out_f)
            with ctx:
                return run_body()
    return run_body()


if __name__ == "__main__":
    raise SystemExit(main())

