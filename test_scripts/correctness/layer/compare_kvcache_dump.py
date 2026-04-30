#!/usr/bin/env python3
"""
Compare KV-cache fingerprint dumps produced by DeepSeek-V2 debug hook
(``_debug_dump_kvcache_fingerprint_if_enabled``) across two runs whose req ids differ.

Dump layout:
  <base>/run_<run_id>/kvcache_fingerprint/kvcache_layer0000_pre_attn_tp0_pos7_reqXXXX.pt

This script:
  - Groups files by req id (from filename), maps reqs across runs by decode pos coverage
    signature, then compares matched keys by (pos, layer_id, tp_rank, stage).
  - If key_sample/val_sample tensors exist, compares them numerically.

Usage:
  python compare_kvcache_dump.py --a /path/to/runA/kvcache_fingerprint --b /path/to/runB/kvcache_fingerprint -o kvcache.diff
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


_FNAME_RE = re.compile(
    r"^kvcache_layer(?P<layer>\d{4})_(?P<stage>.+?)_tp(?P<tp>\d+)_pos(?P<pos>\d+)_req(?P<req>.+?)\.pt$"
)


def _load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass(frozen=True)
class Key:
    pos: int
    layer: int
    stage: str
    tp_rank: int


@dataclass(frozen=True)
class ReqSig:
    min_pos: int
    max_pos: int
    pos_count: int


def _key_sort(k: Key) -> Tuple:
    return (k.pos, k.layer, k.tp_rank, k.stage)


def _iter_files(dir_path: str) -> List[str]:
    out = []
    for fn in os.listdir(dir_path):
        if fn.endswith(".pt") and fn.startswith("kvcache_layer"):
            out.append(os.path.join(dir_path, fn))
    out.sort()
    return out


def _build_req_index(dir_path: str) -> Dict[str, Dict[Key, str]]:
    by_req: Dict[str, Dict[Key, str]] = {}
    for path in _iter_files(dir_path):
        fn = os.path.basename(path)
        m = _FNAME_RE.match(fn)
        if m is None:
            continue
        layer = int(m.group("layer"))
        stage = m.group("stage")
        tp = int(m.group("tp"))
        pos = int(m.group("pos"))
        req = m.group("req")
        k = Key(pos=pos, layer=layer, stage=stage, tp_rank=tp)
        by_req.setdefault(req, {}).setdefault(k, path)
    return by_req


def _req_sig(idx: Dict[Key, str]) -> Optional[ReqSig]:
    poss = sorted({k.pos for k in idx.keys()})
    if not poss:
        return None
    return ReqSig(min_pos=poss[0], max_pos=poss[-1], pos_count=len(poss))


def _build_req_mapping(a: Dict[str, Dict[Key, str]], b: Dict[str, Dict[Key, str]]) -> Tuple[Dict[str, str], List[str]]:
    notes: List[str] = []
    sig_to_a: Dict[ReqSig, List[str]] = {}
    sig_to_b: Dict[ReqSig, List[str]] = {}
    for rid, idx in a.items():
        s = _req_sig(idx)
        if s is not None:
            sig_to_a.setdefault(s, []).append(rid)
    for rid, idx in b.items():
        s = _req_sig(idx)
        if s is not None:
            sig_to_b.setdefault(s, []).append(rid)
    mapping: Dict[str, str] = {}
    common = sorted(set(sig_to_a) & set(sig_to_b), key=lambda s: (s.min_pos, s.max_pos, s.pos_count))
    for s in common:
        la = sorted(sig_to_a[s])
        lb = sorted(sig_to_b[s])
        n = min(len(la), len(lb))
        if len(la) != 1 or len(lb) != 1:
            notes.append(f"signature collision {s}: A={la} B={lb} -> pairing first {n} by sorted order")
        for i in range(n):
            mapping[la[i]] = lb[i]
    return mapping, notes


def _tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if a.shape != b.shape:
        return None
    return float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0


def compare_one(pa: str, pb: str) -> int:
    bad = 0
    da, db = _load(pa), _load(pb)
    # Compare light metadata
    for k in ("pos", "layer_id", "stage", "tp_rank"):
        if da.get(k) != db.get(k):
            print(f"  meta.{k}: A={da.get(k)!r} B={db.get(k)!r} [diff]")
            bad += 1
    # Compare indices summaries
    for k in ("kv_len",):
        if da.get(k) != db.get(k):
            print(f"  {k}: A={da.get(k)!r} B={db.get(k)!r} [diff]")
            bad += 1
    for k in ("kv_indices_head", "kv_indices_tail", "kv_indices_sample"):
        ta, tb = da.get(k), db.get(k)
        if torch.is_tensor(ta) and torch.is_tensor(tb):
            if ta.shape != tb.shape or not torch.equal(ta, tb):
                print(f"  {k}: tensor diff shape A={tuple(ta.shape)} B={tuple(tb.shape)}")
                bad += 1
        else:
            if (ta is None) != (tb is None):
                print(f"  {k}: one missing [diff]")
                bad += 1
    # Compare samples if present
    for k in ("key_sample", "val_sample"):
        ta, tb = da.get(k), db.get(k)
        if torch.is_tensor(ta) and torch.is_tensor(tb):
            if ta.shape != tb.shape:
                print(f"  {k}: SHAPE A={tuple(ta.shape)} B={tuple(tb.shape)} [diff]")
                bad += 1
            else:
                m = _tensor_max_abs(ta, tb)
                print(f"  {k}: max_abs={m:.6g}")
                if m is not None and m > 0:
                    bad += 1
        elif ta is None and tb is None:
            print(f"  {k}: both None")
        else:
            print(f"  {k}: one missing [diff]")
            bad += 1
    return bad


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="Run A kvcache_fingerprint dir")
    p.add_argument("--b", required=True, help="Run B kvcache_fingerprint dir")
    p.add_argument("-o", "--output", default="", help="Write report to file (and tee to stdout if TTY)")
    p.add_argument("--limit", type=int, default=0, help="Compare at most N file pairs")
    args = p.parse_args(list(argv) if argv is not None else None)

    dir_a = os.path.abspath(os.path.expanduser(args.a))
    dir_b = os.path.abspath(os.path.expanduser(args.b))
    if not os.path.isdir(dir_a):
        print(f"ERROR: not a directory: {dir_a}", file=sys.stderr)
        return 2
    if not os.path.isdir(dir_b):
        print(f"ERROR: not a directory: {dir_b}", file=sys.stderr)
        return 2

    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else ""

    def run_body() -> int:
        req_a = _build_req_index(dir_a)
        req_b = _build_req_index(dir_b)
        mapping, notes = _build_req_mapping(req_a, req_b)
        print(f"A: {dir_a}")
        print(f"B: {dir_b}")
        print(f"A req groups: {len(req_a)}")
        print(f"B req groups: {len(req_b)}")
        for n in notes:
            print(f"# {n}")
        print(f"mapped reqs: {len(mapping)}")
        print()

        compared = 0
        bad = 0
        for ra, rb in sorted(mapping.items()):
            ia = req_a.get(ra, {})
            ib = req_b.get(rb, {})
            common = sorted(set(ia.keys()) & set(ib.keys()), key=_key_sort)
            print(f"== req pair A={ra} B={rb}: common_keys={len(common)} ==")
            for k in common:
                print(f"=== key pos={k.pos} layer={k.layer} tp={k.tp_rank} stage={k.stage} ===")
                pa, pb = ia[k], ib[k]
                print(f"A: {pa}")
                print(f"B: {pb}")
                try:
                    bad += compare_one(pa, pb)
                except Exception as e:
                    print(f"  COMPARE_ERROR: {e}")
                    bad += 1
                print()
                compared += 1
                if args.limit and compared >= args.limit:
                    print("(limit reached)")
                    print("--- summary ---")
                    print(f"compared: {compared}")
                    print(f"bad score: {bad}")
                    return 0 if bad == 0 else 1
        print("--- summary ---")
        print(f"compared: {compared}")
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

