#!/usr/bin/env python3
"""
Compare **decode** ``mlpinner_*`` checkpoints from ``SGLANG_DEBUG_DUMP_MLP_INNER``
(``_debug_save_mlp_inner_if_enabled`` in ``deepseek_v2.py``).

``compare_decode_layer_hiddens.py`` only matches filenames ``layer_*_decode_*``;
it **never** sees ``mlpinner_*``. Use this script for MLP/MoE inner tensors + meta.

Expected filenames (typical single-token decode):
  mlpinner_layer_{layer:04d}_{stage}_tp{tp}_decode_pos{pos}_req{...}.pt

Usage:
  python compare_decode_mlp_inner.py \\
    --a /path/run_A/normal_forward \\
    --b /path/run_B/normal_forward \\
    -o run_ab.mlpinner.diff

  python compare_decode_mlp_inner.py --a ... --b ... --only-stages moe_router_meta,dense_act
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TextIO, Tuple

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


_MLPINNER_RE = re.compile(
    r"^mlpinner_layer_(?P<layer>\d{4})_(?P<stage>.+?)_tp(?P<tp>\d+)"
    r"_decode_(?P<dbody>.+?)_req(?P<req>.+?)\.pt$"
)
_POS_SINGLE = re.compile(r"^pos(\d+)$")


@dataclass(frozen=True)
class InnerKey:
    layer: int
    stage: str
    tp_rank: int
    pos: int  # -1 if decode suffix is not ``pos{N}`` (e.g. multi-token ``npos...``)
    decode_body: str


@dataclass(frozen=True)
class ReqSig:
    decode_min_pos: int
    decode_max_pos: int
    decode_pos_count: int


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


def _decode_pos_from_mlpinner_basename(fn: str) -> Optional[int]:
    m = _MLPINNER_RE.match(fn)
    if not m:
        return None
    dbody = m.group("dbody")
    pm = _POS_SINGLE.match(dbody)
    if pm:
        return int(pm.group(1))
    return None


def _inner_key_from_basename(fn: str) -> Optional[Tuple[InnerKey, str]]:
    m = _MLPINNER_RE.match(fn)
    if m is None:
        return None
    layer = int(m.group("layer"))
    stage = m.group("stage")
    tp = int(m.group("tp"))
    dbody = m.group("dbody")
    req = m.group("req")
    pm = _POS_SINGLE.match(dbody)
    pos = int(pm.group(1)) if pm else -1
    k = InnerKey(layer=layer, stage=stage, tp_rank=tp, pos=pos, decode_body=dbody)
    return k, req


def _iter_mlpinner_decode_files(dir_path: str) -> List[str]:
    out: List[str] = []
    try:
        names = os.listdir(dir_path)
    except FileNotFoundError:
        return []
    for fn in names:
        if not fn.startswith("mlpinner_") or not fn.endswith(".pt"):
            continue
        if "_decode_" not in fn:
            continue
        out.append(os.path.join(dir_path, fn))
    out.sort()
    return out


def _req_sig_from_paths(paths: Iterable[str]) -> Optional[ReqSig]:
    poss: List[int] = []
    for p in paths:
        fn = os.path.basename(p)
        pos = _decode_pos_from_mlpinner_basename(fn)
        if pos is not None:
            poss.append(pos)
    if not poss:
        return None
    poss.sort()
    uniq = sorted(set(poss))
    return ReqSig(
        decode_min_pos=uniq[0],
        decode_max_pos=uniq[-1],
        decode_pos_count=len(uniq),
    )


def _build_req_index(dir_path: str) -> Dict[str, Dict[InnerKey, str]]:
    by_req: Dict[str, Dict[InnerKey, str]] = {}
    for path in _iter_mlpinner_decode_files(dir_path):
        fn = os.path.basename(path)
        parsed = _inner_key_from_basename(fn)
        if parsed is None:
            continue
        k, req = parsed
        by_req.setdefault(req, {})[k] = path
    return by_req


def _build_req_mapping(
    reqs_a: Dict[str, Dict[InnerKey, str]],
    reqs_b: Dict[str, Dict[InnerKey, str]],
) -> Tuple[Dict[str, str], List[str]]:
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


def _meta_line_diff(ma: Any, mb: Any, indent: str = "  ") -> List[str]:
    lines: List[str] = []
    if type(ma) != type(mb):
        lines.append(f"{indent}type diff: A={type(ma).__name__} B={type(mb).__name__}")
        return lines
    if ma == mb:
        return lines
    if isinstance(ma, dict) and isinstance(mb, dict):
        keys_a, keys_b = set(ma), set(mb)
        for k in sorted(keys_a - keys_b):
            lines.append(f"{indent}meta only A: {k!r}")
        for k in sorted(keys_b - keys_a):
            lines.append(f"{indent}meta only B: {k!r}")
        for k in sorted(keys_a & keys_b):
            sub = _meta_line_diff(ma[k], mb[k], indent + "  ")
            if sub:
                lines.append(f"{indent}meta[{k!r}]:")
                lines.extend(sub)
        return lines
    lines.append(f"{indent}value A={ma!r} B={mb!r}")
    return lines


def _inner_key_sort(k: InnerKey) -> Tuple:
    stage_order = {
        "moe_router_meta": 0,
        "moe_routed_maybe_scaled_pre_shared": 1,
        "moe_pre_tp_allreduce": 2,
        "dense_gate_up": 3,
        "dense_act": 4,
        "dense_down_proj": 5,
    }
    return (k.pos if k.pos >= 0 else 10**9, k.layer, stage_order.get(k.stage, 99), k.tp_rank, k.stage)


def compare_pair(
    key: InnerKey,
    pa: str,
    pb: str,
    *,
    strict: bool,
    cos_tol: float,
) -> int:
    bad = 0
    print(
        f"=== mlpinner key: pos={key.pos} layer={key.layer} tp={key.tp_rank} "
        f"stage={key.stage} decode_body={key.decode_body!r} ==="
    )
    print(f"A: {pa}")
    print(f"B: {pb}")
    try:
        da, db = _load(pa), _load(pb)
    except Exception as e:
        print(f"  LOAD_ERROR: {e}")
        return 1

    for mk in ("dump_kind", "layer_id", "stage", "decode_dump", "tp_rank", "attn_dp_rank", "forward_mode"):
        va, vb = da.get(mk, None), db.get(mk, None)
        if va != vb:
            print(f"  payload.{mk}: A={va!r} B={vb!r}  [diff]")

    ma, mb = da.get("meta") or {}, db.get("meta") or {}
    mdiff = _meta_line_diff(ma, mb)
    if mdiff:
        print("  meta diff:")
        for line in mdiff:
            print(line)

    for tk in ("tensor", "residual"):
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
        line, m = _tensor_stats(ta2, tb2, tk)
        print(line)
        if strict and m is not None and _strict_fail(m, cos_tol):
            bad += 1
    print()
    return bad


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="Run A normal_forward dir")
    p.add_argument("--b", required=True, help="Run B normal_forward dir")
    p.add_argument("-o", "--output", default="", help="Write report to file (tee if TTY)")
    p.add_argument("--strict", action="store_true", help="Non-zero exit if tensor cos tolerance fails")
    p.add_argument("--cos-tol", type=float, default=1e-5)
    p.add_argument(
        "--only-stages",
        default="",
        help="Comma-separated inner stage names (e.g. moe_router_meta,dense_act); empty = all",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    dir_a = os.path.abspath(os.path.expanduser(args.a))
    dir_b = os.path.abspath(os.path.expanduser(args.b))
    if not os.path.isdir(dir_a):
        print(f"ERROR: not a directory: {dir_a}", file=sys.stderr)
        return 2
    if not os.path.isdir(dir_b):
        print(f"ERROR: not a directory: {dir_b}", file=sys.stderr)
        return 2

    only_stages: Optional[Set[str]] = None
    if (args.only_stages or "").strip():
        only_stages = {s.strip() for s in args.only_stages.split(",") if s.strip()}

    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else ""

    def run_body() -> int:
        req_a = _build_req_index(dir_a)
        req_b = _build_req_index(dir_b)
        mapping, notes = _build_req_mapping(req_a, req_b)
        print(f"A: {dir_a}")
        print(f"B: {dir_b}")
        print(f"A mlpinner decode files: {len(_iter_mlpinner_decode_files(dir_a))}")
        print(f"B mlpinner decode files: {len(_iter_mlpinner_decode_files(dir_b))}")
        print(f"A req groups: {len(req_a)}")
        print(f"B req groups: {len(req_b)}")
        for n in notes:
            print(f"# {n}")
        print(f"mapped reqs: {len(mapping)}")
        if not mapping:
            print(
                "\n# No request mapping — check that both dirs contain mlpinner_*_decode_*.pt "
                "and overlapping decode position signatures.",
            )
        print()

        bad = 0
        compared = 0
        for ra, rb in sorted(mapping.items()):
            ia, ib = req_a.get(ra, {}), req_b.get(rb, {})
            keys_a, keys_b = set(ia.keys()), set(ib.keys())
            common = sorted(keys_a & keys_b, key=_inner_key_sort)
            if only_stages is not None:
                common = [k for k in common if k.stage in only_stages]

            tp_map: Dict[int, int] = {}
            if not common:
                tpa = sorted({k.tp_rank for k in keys_a})
                tpb = sorted({k.tp_rank for k in keys_b})
                if len(tpa) == 1 and len(tpb) == 1 and tpa[0] != tpb[0]:
                    tp_map[tpa[0]] = tpb[0]
                    print(
                        f"== req pair A={ra} B={rb}: no direct key match, tp_rank remap "
                        f"A{tpa[0]}->B{tpb[0]} =="
                    )
                    common2: List[InnerKey] = []
                    for ka in keys_a:
                        if only_stages is not None and ka.stage not in only_stages:
                            continue
                        kb = InnerKey(
                            layer=ka.layer,
                            stage=ka.stage,
                            tp_rank=tp_map.get(ka.tp_rank, ka.tp_rank),
                            pos=ka.pos,
                            decode_body=ka.decode_body,
                        )
                        if kb in keys_b:
                            common2.append(ka)
                    common = sorted(common2, key=_inner_key_sort)
                    print(f"== after tp remap: common_keys={len(common)} ==")

            if not common:
                print(f"== req pair A={ra} B={rb}: no common mlpinner keys ==")
                bad += 1
                continue

            print(f"== req pair A={ra} B={rb}: common_keys={len(common)} ==")

            for k in common:
                kb = InnerKey(
                    layer=k.layer,
                    stage=k.stage,
                    tp_rank=tp_map.get(k.tp_rank, k.tp_rank),
                    pos=k.pos,
                    decode_body=k.decode_body,
                )
                pa = ia[k]
                pb_path = ib.get(kb)
                if pb_path is None:
                    print(f"  missing B for key {k} (lookup {kb})")
                    bad += 1
                    continue
                bad += compare_pair(k, pa, pb_path, strict=args.strict, cos_tol=args.cos_tol)
                compared += 1

        print("--- summary ---")
        print(f"req pairs: {len(mapping)}")
        print(f"compared mlpinner keys: {compared}")
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
