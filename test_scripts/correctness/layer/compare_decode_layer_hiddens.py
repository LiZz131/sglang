#!/usr/bin/env python3
"""
Compare **decode** per-layer hidden dumps produced by DeepSeek-V2 debug hook
(``_debug_save_layer_hidden_if_enabled``) across two runs whose req ids differ.

This script matches files by a **stable signature** derived from filename +
payload, rather than relying on request id strings.

Expected filenames (decode-enabled):
  layer_{layer:04d}_{stage}_tp{tp}_decode_pos{pos}_req{...}.pt

Usage:
  python compare_decode_layer_hiddens.py \
    --a /path/to/run_..._tp/normal_forward \
    --b /path/to/run_..._dpattn/normal_forward \
    -o run_0430_decode.hidden.decode.diff

Notes:
  - Only compares files whose name contains ``_decode_``.
  - Does NOT use sampled token ids (input_ids) as matching keys.
  - Maps request ids across runs by decode pos coverage, then matches tensors by
    (pos, layer_id, stage, tp_rank) within each mapped request.
  - Tensors compared: payload["hidden_states"] and payload["residual"] (if present).
"""

from __future__ import annotations

import argparse
import contextlib
import math
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
    r"^layer_(?P<layer>\d{4})_(?P<stage>.+?)_tp(?P<tp>\d+)_decode_pos(?P<pos>\d+)_req(?P<req>.+?)\.pt$"
)


def _load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_stats(a: torch.Tensor, b: torch.Tensor, name: str) -> Tuple[str, Optional[Dict[str, float]]]:
    if a.shape != b.shape:
        return (
            f"  {name}: SHAPE_MISMATCH  a={tuple(a.shape)}  b={tuple(b.shape)}",
            None,
        )
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    rmse = float(torch.sqrt(torch.mean((af - bf) ** 2)).item()) if diff.numel() else 0.0
    denom = float(torch.norm(af).item())
    rel = rmse / denom if denom > 1e-12 else float("nan")
    cos = float(
        (af.flatten() * bf.flatten()).sum().item()
        / (af.flatten().norm().item() * bf.flatten().norm().item() + 1e-12)
    )
    line = (
        f"  {name}: max_abs={max_abs:.6g}  mean_abs={mean_abs:.6g}  "
        f"rmse={rmse:.6g}  rel_rmse={rel:.6g}  cos={cos:.8f}"
    )
    return line, {"max_abs": max_abs, "cos": cos}


def _strict_fail(metrics: Dict[str, float], cos_tol: float) -> bool:
    return metrics["max_abs"] > 0 and (
        not math.isfinite(metrics["cos"]) or abs(1.0 - metrics["cos"]) > cos_tol
    )

def _best_row_match_by_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    name: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Conservative row matching when batch differs.

    If A is (1, ...) and B is (N, ...) with same trailing dims, compare A[0] with each
    B[i] and choose the row that is *closest* by (max cos, then min max_abs).
    Symmetric when B is (1, ...) and A is (N, ...).
    """
    notes: List[str] = []
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return a, b, notes
    if a.dim() < 1 or b.dim() < 1:
        return a, b, notes
    if a.shape == b.shape:
        return a, b, notes
    if a.shape[1:] != b.shape[1:]:
        return a, b, notes
    if a.shape[0] != 1 and b.shape[0] != 1:
        return a, b, notes

    def pick_one(ref: torch.Tensor, cand: torch.Tensor, cand_label: str) -> Tuple[int, float, float]:
        best_i = 0
        best_cos = -1.0
        best_max_abs = float("inf")
        for i in range(int(cand.shape[0])):
            _line, m = _tensor_stats(ref, cand[i : i + 1], f"{name}[{cand_label}{i}]")
            if m is None:
                continue
            cos = float(m["cos"])
            max_abs = float(m["max_abs"])
            if cos > best_cos + 1e-12 or (abs(cos - best_cos) <= 1e-12 and max_abs < best_max_abs):
                best_cos, best_max_abs, best_i = cos, max_abs, i
        return best_i, best_cos, best_max_abs

    if a.shape[0] == 1 and b.shape[0] > 1:
        i, cos, max_abs = pick_one(a[0:1], b, "B")
        notes.append(
            f"{name}: batch_mismatch A=1 B={int(b.shape[0])} -> choose B[{i}] (cos={cos:.6g}, max_abs={max_abs:.6g})"
        )
        return a[0:1], b[i : i + 1], notes
    if b.shape[0] == 1 and a.shape[0] > 1:
        i, cos, max_abs = pick_one(b[0:1], a, "A")
        notes.append(
            f"{name}: batch_mismatch A={int(a.shape[0])} B=1 -> choose A[{i}] (cos={cos:.6g}, max_abs={max_abs:.6g})"
        )
        return a[i : i + 1], b[0:1], notes
    return a, b, notes


@dataclass(frozen=True)
class Key:
    layer: int
    stage: str
    tp_rank: int
    pos: int


@dataclass(frozen=True)
class ReqSig:
    decode_min_pos: int
    decode_max_pos: int
    decode_pos_count: int


def _key_sort(k: Key) -> Tuple:
    return (k.pos, k.layer, k.tp_rank, k.stage)


def _req_sig_from_paths(paths: Iterable[str]) -> Optional[ReqSig]:
    poss: List[int] = []
    for p in paths:
        fn = os.path.basename(p)
        m = _FNAME_RE.match(fn)
        if m is not None:
            poss.append(int(m.group("pos")))
    if not poss:
        return None
    poss.sort()
    return ReqSig(
        decode_min_pos=poss[0],
        decode_max_pos=poss[-1],
        decode_pos_count=len(set(poss)),
    )


def _iter_decode_files(dir_path: str) -> List[str]:
    files = []
    for fn in os.listdir(dir_path):
        if not fn.endswith(".pt"):
            continue
        if "_decode_" not in fn:
            continue
        files.append(os.path.join(dir_path, fn))
    files.sort()
    return files


def _build_req_index(dir_path: str) -> Dict[str, Dict[Key, str]]:
    """Extract per-request indices from filenames.

    Returns: req_id_in_filename -> (Key -> path)
    """
    by_req: Dict[str, Dict[Key, str]] = {}
    for path in _iter_decode_files(dir_path):
        fn = os.path.basename(path)
        m = _FNAME_RE.match(fn)
        if m is None:
            continue
        layer = int(m.group("layer"))
        stage = m.group("stage")
        tp = int(m.group("tp"))
        pos = int(m.group("pos"))
        req = m.group("req")
        k = Key(layer=layer, stage=stage, tp_rank=tp, pos=pos)
        by_req.setdefault(req, {}).setdefault(k, path)
    return by_req


def _build_req_mapping(
    reqs_a: Dict[str, Dict[Key, str]],
    reqs_b: Dict[str, Dict[Key, str]],
) -> Tuple[Dict[str, str], List[str]]:
    """Map request ids across runs using decode pos coverage signature."""
    notes: List[str] = []
    sig_to_a: Dict[ReqSig, List[str]] = {}
    sig_to_b: Dict[ReqSig, List[str]] = {}

    for rid, idx in reqs_a.items():
        sig = _req_sig_from_paths(idx.values())
        if sig is not None:
            sig_to_a.setdefault(sig, []).append(rid)
    for rid, idx in reqs_b.items():
        sig = _req_sig_from_paths(idx.values())
        if sig is not None:
            sig_to_b.setdefault(sig, []).append(rid)

    mapping: Dict[str, str] = {}
    common_sigs = sorted(
        set(sig_to_a) & set(sig_to_b),
        key=lambda s: (s.decode_min_pos, s.decode_max_pos, s.decode_pos_count),
    )
    for sig in common_sigs:
        la = sorted(sig_to_a[sig])
        lb = sorted(sig_to_b[sig])
        n = min(len(la), len(lb))
        if len(la) != 1 or len(lb) != 1:
            notes.append(
                f"signature collision {sig}: A={la} B={lb} -> pairing first {n} by sorted order"
            )
        for i in range(n):
            mapping[la[i]] = lb[i]
    return mapping, notes


def compare_pair(
    key: Key,
    pa: str,
    pb: str,
    *,
    strict: bool,
    cos_tol: float,
) -> int:
    bad = 0
    print(f"=== decode key: pos={key.pos} layer={key.layer} tp={key.tp_rank} stage={key.stage} ===")
    print(f"A: {pa}")
    print(f"B: {pb}")
    try:
        da, db = _load(pa), _load(pb)
    except Exception as e:
        print(f"  LOAD_ERROR: {e}")
        return 1

    # Quick meta sanity
    for mk in ("forward_mode", "tp_rank", "attn_dp_rank", "decode_dump"):
        va, vb = da.get(mk, None), db.get(mk, None)
        if va != vb:
            print(f"  meta.{mk}: A={va!r} B={vb!r}  [diff]")
        else:
            print(f"  meta.{mk}: A==B  ({va!r})")

    for tk in ("hidden_states", "residual"):
        ta, tb = da.get(tk, None), db.get(tk, None)
        if ta is None and tb is None:
            print(f"  {tk}: both None  [ok]")
            continue
        if ta is None or tb is None:
            print(f"  {tk}: one is None  A={ta is not None} B={tb is not None}  [diff]")
            bad += 1
            continue
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            print(f"  {tk}: not tensors  [diff]")
            bad += 1
            continue

        ta2, tb2, notes = _best_row_match_by_metrics(ta, tb, name=tk)
        for n in notes:
            print(f"  {n}")
        ta, tb = ta2, tb2

        line, m = _tensor_stats(ta, tb, tk)
        print(line)
        if strict and m is not None and _strict_fail(m, cos_tol):
            bad += 1
    print()
    return bad


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="Run A normal_forward dir (e.g. TP)")
    p.add_argument("--b", required=True, help="Run B normal_forward dir (e.g. special-dp-attn)")
    p.add_argument("-o", "--output", default="", help="Write report to file (and tee to stdout if TTY)")
    p.add_argument("--strict", action="store_true", help="Non-zero exit if any tensor fails cos tolerance")
    p.add_argument("--cos-tol", type=float, default=1e-5)
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

        bad = 0
        compared = 0
        for ra, rb in sorted(mapping.items()):
            ia = req_a.get(ra, {})
            ib = req_b.get(rb, {})
            keys_a, keys_b = set(ia.keys()), set(ib.keys())
            common = sorted(keys_a & keys_b, key=_key_sort)
            if not common:
                print(f"== req pair A={ra} B={rb}: no common decode keys ==")
                bad += 1
                continue
            print(f"== req pair A={ra} B={rb}: common_keys={len(common)} ==")
            for k in common:
                bad += compare_pair(
                    k,
                    ia[k],
                    ib[k],
                    strict=args.strict,
                    cos_tol=args.cos_tol,
                )
                compared += 1
        print("--- summary ---")
        print(f"req pairs: {len(mapping)}")
        print(f"compared: {compared} key(s)")
        print(f"bad score: {bad}")
        if args.strict and bad > 0:
            return 1
        return 0

    if out_path:
        with open(out_path, "w", encoding="utf-8") as out_f:
            redirect_ctx: Any
            if sys.stdout.isatty():
                redirect_ctx = contextlib.redirect_stdout(_TeeStdout(sys.stdout, out_f))
            else:
                redirect_ctx = contextlib.redirect_stdout(out_f)
            with redirect_ctx:
                return run_body()
    return run_body()


if __name__ == "__main__":
    raise SystemExit(main())

