#!/usr/bin/env python3
"""
Compare decode ``attninner_*`` payloads from ``SGLANG_DEBUG_DUMP_ATTN_INNER=1``
(see ``deepseek_v2.py`` ``_debug_save_attn_inner_if_enabled``).

Filenames mirror ``mlpinner_*`` layout:

  attninner_layer_{layer:04d}_{stage}_tp{tp}_decode_{dbody}_req{...}.pt

Payload holds ``meta`` + ``probes`` (per-tensor flattened samples with stats /
head/tail), not full activations — comparison is probe-level (cos / max_abs).

``meta`` includes MLA absorb routing when present (e.g. ``absorb_core_branch_id``,
``absorb_core_routing``, ``attn_mqa_call``, ``attn_backend_cls``): the script prints
summary lines and a recursive ``meta diff`` when values differ. Use this to verify
both runs took the same ``forward_absorb_core`` branch and ``forward_batch.attn_backend``
class before interpreting full-tensor diffs.

Typical stages (subset may exist per forward path): ``attn_dispatch``,
``MHA_qkv``, ``MLA_post_mqa``, ``attn_inout``.

**Output:** default is **compact** (basename header, probe lines only when head/tail
pairwise drift is noticeable or ``--strict`` fails). Use ``--v`` / ``--verbose`` for
full per-stat lines and absolute paths. **Progress:** ``tqdm`` on stderr when available
and stderr is a TTY (``--no-progress`` to disable).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

from compare_decode_mlp_inner import (  # noqa: E402
    InnerKey,
    ReqSig,
    _TeeStdout,
    _best_row_match_by_metrics,
    _meta_line_diff,
    _tensor_stats,
    _strict_fail,
)

_ATTNINNER_RE = re.compile(
    r"^attninner_layer_(?P<layer>\d{4})_(?P<stage>.+?)_tp(?P<tp>\d+)"
    r"_decode_(?P<dbody>.+?)_req(?P<req>.+?)\.pt$"
)
_POS_SINGLE = re.compile(r"^pos(\d+)$")


def _iter_attninner_decode_files(dir_path: str) -> List[str]:
    out: List[str] = []
    try:
        names = os.listdir(dir_path)
    except FileNotFoundError:
        return []
    for fn in names:
        if not fn.startswith("attninner_") or not fn.endswith(".pt"):
            continue
        if "_decode_" not in fn:
            continue
        out.append(os.path.join(dir_path, fn))
    out.sort()
    return out


def _decode_pos_from_attninner_basename(fn: str) -> Optional[int]:
    m = _ATTNINNER_RE.match(fn)
    if not m:
        return None
    dbody = m.group("dbody")
    pm = _POS_SINGLE.match(dbody)
    if pm:
        return int(pm.group(1))
    return None


def _inner_key_attn_from_basename(fn: str) -> Optional[Tuple[InnerKey, str]]:
    m = _ATTNINNER_RE.match(fn)
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


def _req_sig_from_attn_paths(paths: List[str]) -> Optional[ReqSig]:
    poss: List[int] = []
    for p in paths:
        fn = os.path.basename(p)
        pos = _decode_pos_from_attninner_basename(fn)
        if pos is not None:
            poss.append(pos)
    if not poss:
        return None
    uniq = sorted(set(poss))
    return ReqSig(
        decode_min_pos=uniq[0],
        decode_max_pos=uniq[-1],
        decode_pos_count=len(uniq),
    )


def _build_req_index(dir_path: str) -> Dict[str, Dict[InnerKey, str]]:
    by_req: Dict[str, Dict[InnerKey, str]] = {}
    for path in _iter_attninner_decode_files(dir_path):
        fn = os.path.basename(path)
        parsed = _inner_key_attn_from_basename(fn)
        if parsed is None:
            continue
        k, req = parsed
        by_req.setdefault(req, {})[k] = path
    return by_req


def _build_req_mapping_attn(
    reqs_a: Dict[str, Dict[InnerKey, str]],
    reqs_b: Dict[str, Dict[InnerKey, str]],
) -> Tuple[Dict[str, str], List[str]]:
    """Same signature-based pairing as mlpinner, using attninner paths."""

    notes: List[str] = []
    sig_to_a: Dict[ReqSig, List[str]] = {}
    sig_to_b: Dict[ReqSig, List[str]] = {}

    for rid, idx in reqs_a.items():
        sig = _req_sig_from_attn_paths(list(idx.values()))
        if sig is not None:
            sig_to_a.setdefault(sig, []).append(rid)
    for rid, idx in reqs_b.items():
        sig = _req_sig_from_attn_paths(list(idx.values()))
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
                f"(attninner) signature collision {sig}: A={la} B={lb} "
                f"-> pairing first {n} by sorted order"
            )
        for i in range(n):
            mapping[la[i]] = lb[i]
    return mapping, notes


def _pair_list_metrics(a: List[float], b: List[float]) -> Dict[str, float]:
    """Pairwise A vs B on same-length lists (float64 path)."""

    if len(a) != len(b) or not a:
        return {
            "cos": float("nan"),
            "max_abs": float("inf"),
            "rmse": float("nan"),
            "l2": float("nan"),
            "max_rel": float("nan"),
        }
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    diff = ta - tb
    n1 = float(ta.norm().item())
    n2 = float(tb.norm().item())
    cos = float((ta * tb).sum().item() / (n1 * n2 + 1e-12))
    return {
        "cos": cos,
        "max_abs": float(diff.abs().max().item()),
        "rmse": float(torch.sqrt((diff**2).mean()).item()),
        "l2": float(torch.norm(diff).item()),
        "max_rel": float((diff.abs() / (ta.abs() + 1e-30)).max().item()),
    }


def _pick_head_tail_lists(
    sa: Dict[str, Any], sb: Dict[str, Any]
) -> Tuple[List[float], List[float], str, List[float], List[float], str]:
    """Prefer fp64 head/tail when both sides have them (newer dumps)."""

    ha64, hb64 = sa.get("head_f64"), sb.get("head_f64")
    if isinstance(ha64, list) and isinstance(hb64, list) and ha64 and hb64:
        ta64, tb64 = sa.get("tail_f64"), sb.get("tail_f64")
        if not (isinstance(ta64, list) and isinstance(tb64, list) and ta64 and tb64):
            ta64, tb64 = sa.get("tail_f32"), sb.get("tail_f32")
        if isinstance(ta64, list) and isinstance(tb64, list) and ta64 and tb64:
            return ha64, hb64, "fp64", ta64, tb64, "fp64"
    ha = sa.get("head_f32") or []
    hb = sb.get("head_f32") or []
    ta = sa.get("tail_f32") or []
    tb = sb.get("tail_f32") or []
    return ha, hb, "fp32", ta, tb, "fp32"


def _compare_one_probe(
    name: str,
    pa: Any,
    pb: Any,
    *,
    strict: bool,
    cos_tol: float,
    compact: bool = False,
) -> Tuple[int, List[str]]:
    """Return (bad_increment, lines)."""

    lines: List[str] = []
    bad = 0
    if pa is None and pb is None:
        if not compact:
            lines.append(f"  probe[{name}]: both absent  [ok]")
        return bad, lines
    if pa is None or pb is None:
        lines.append(
            f"  probe[{name}]: one absent  A={pa is not None} B={pb is not None}  [diff]"
        )
        return 1, lines
    if not isinstance(pa, dict) or not isinstance(pb, dict):
        lines.append(f"  probe[{name}]: non-dict payload  [diff]")
        return 1, lines

    if pa.get("present") != pb.get("present"):
        lines.append(
            f"  probe[{name}]: present A={pa.get('present')} B={pb.get('present')}  [diff]"
        )
        bad += 1
    sh_a, sh_b = pa.get("shape"), pb.get("shape")
    if sh_a != sh_b:
        lines.append(f"  probe[{name}]: shape A={sh_a} B={sh_b}  [diff]")
        bad += 1

    sa, sb = pa.get("stats") or {}, pb.get("stats") or {}
    if not isinstance(sa, dict) or not isinstance(sb, dict):
        lines.append(f"  probe[{name}]: stats not dict  [diff]")
        return bad + 1, lines

    if not compact:
        for sk in ("max_abs", "mean", "std"):
            va, vb = sa.get(sk), sb.get(sk)
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                lines.append(f"  probe[{name}].stats.{sk}: one missing  [diff]")
                bad += 1
                continue
            dv = abs(float(va) - float(vb))
            lines.append(
                f"  probe[{name}].stats.{sk}: A={va:.6g} B={vb:.6g} delta={dv:.6g}  "
                f"(each run's own flat-sample stat, not A−B elementwise)"
            )

    ha, hb, hband, taa, tbb, tband = _pick_head_tail_lists(sa, sb)
    head_strict_bad = False
    if isinstance(ha, list) and isinstance(hb, list) and ha and hb:
        mh = _pair_list_metrics(ha, hb)
        cos_h = mh["cos"]
        mx_h = mh["max_abs"]
        if not compact:
            lines.append(
                f"  probe[{name}].stats.head_{hband}: len={len(ha)} pairwise_A_vs_B "
                f"max_abs={mx_h:.10e} rmse={mh['rmse']:.10e} l2={mh['l2']:.10e} "
                f"max_rel={mh['max_rel']:.10e} cos={cos_h:.10f}"
            )
            if len(ha) <= 8 and mx_h < 1e-12 and cos_h < 1.0 - 1e-6:
                d0 = [float(ha[i]) - float(hb[i]) for i in range(len(ha))]
                lines.append(f"    (sanity) raw_delta_head={d0!r}")
        else:
            noisy_h = (
                mx_h > 1e-10
                or cos_h < 1.0 - max(cos_tol * 10, 1e-6)
                or math.isnan(cos_h)
            )
            if strict and (
                math.isnan(cos_h)
                or (len(ha) == len(hb) and _strict_fail({"max_abs": mx_h, "cos": cos_h}, cos_tol))
            ):
                head_strict_bad = True
            if noisy_h or head_strict_bad:
                tag = " [strict]" if head_strict_bad else ""
                lines.append(
                    f"  probe[{name}] head_{hband}: max_abs={mx_h:.10e} cos={cos_h:.10f}{tag}"
                )
        if strict and (
            math.isnan(cos_h)
            or (len(ha) == len(hb) and _strict_fail({"max_abs": mx_h, "cos": cos_h}, cos_tol))
        ):
            bad += 1
    elif (ha or hb) and type(ha) != type(hb):
        lines.append(f"  probe[{name}].stats.head: type/shape mismatch  [diff]")
        bad += 1

    if isinstance(taa, list) and isinstance(tbb, list) and taa and tbb:
        mt = _pair_list_metrics(taa, tbb)
        cos_t = mt["cos"]
        mx_t = mt["max_abs"]
        if not compact:
            lines.append(
                f"  probe[{name}].stats.tail_{tband}: len={len(taa)} pairwise_A_vs_B "
                f"max_abs={mx_t:.10e} rmse={mt['rmse']:.10e} l2={mt['l2']:.10e} "
                f"max_rel={mt['max_rel']:.10e} cos={cos_t:.10f}"
            )
        else:
            noisy_t = (
                mx_t > 1e-10
                or cos_t < 1.0 - max(cos_tol * 10, 1e-6)
                or math.isnan(cos_t)
            )
            tail_strict_bad = bool(
                strict
                and (
                    math.isnan(cos_t)
                    or (
                        len(taa) == len(tbb)
                        and _strict_fail({"max_abs": mx_t, "cos": cos_t}, cos_tol)
                    )
                )
            )
            if noisy_t or tail_strict_bad:
                tag = " [strict]" if tail_strict_bad else ""
                lines.append(
                    f"  probe[{name}] tail_{tband}: max_abs={mx_t:.10e} cos={cos_t:.10f}{tag}"
                )
        if strict and (
            math.isnan(cos_t)
            or (
                len(taa) == len(tbb)
                and _strict_fail({"max_abs": mx_t, "cos": cos_t}, cos_tol)
            )
        ):
            bad += 1
    elif (taa or tbb) and type(taa) != type(tbb):
        lines.append(f"  probe[{name}].stats.tail: type mismatch  [diff]")
        bad += 1

    return bad, lines


def compare_pair(
    key: InnerKey,
    pa: str,
    pb: str,
    *,
    strict: bool,
    cos_tol: float,
    verbose: bool,
) -> int:
    bad = 0
    compact = not verbose
    if verbose:
        print(
            f"=== attninner key: pos={key.pos} layer={key.layer} tp={key.tp_rank} "
            f"stage={key.stage} decode_body={key.decode_body!r} ==="
        )
        print(f"A: {pa}")
        print(f"B: {pb}")
    else:
        print(
            f"=== pos={key.pos} layer={key.layer} tp={key.tp_rank} stage={key.stage} "
            f"| {os.path.basename(pa)} | {os.path.basename(pb)} ==="
        )
    try:
        da, db = torch.load(pa, map_location="cpu", weights_only=False), torch.load(
            pb, map_location="cpu", weights_only=False
        )
    except TypeError:
        da, db = torch.load(pa, map_location="cpu"), torch.load(pb, map_location="cpu")
    except Exception as e:
        print(f"  LOAD_ERROR: {e}")
        return 1

    for mk in ("dump_kind", "layer_id", "stage", "decode_dump", "tp_rank", "attn_dp_rank"):
        va, vb = da.get(mk, None), db.get(mk, None)
        if va != vb:
            print(f"  payload.{mk}: A={va!r} B={vb!r}  [diff]")

    ma, mb = da.get("meta") or {}, db.get("meta") or {}
    br_a, br_b = ma.get("absorb_core_branch_id"), mb.get("absorb_core_branch_id")
    if br_a is not None or br_b is not None:
        br_tag = " [diff]" if br_a != br_b else " (match)"
        print(
            f"  absorb_core_branch_id: A={br_a!r} B={br_b!r}{br_tag}  "
            f"routing A={ma.get('absorb_core_routing')!r} B={mb.get('absorb_core_routing')!r}  "
            f"attn_call A={ma.get('attn_mqa_call')!r} B={mb.get('attn_mqa_call')!r}  "
            f"layout A={ma.get('attn_mqa_tensor_layout')!r} B={mb.get('attn_mqa_tensor_layout')!r}"
        )
    cls_a, cls_b = ma.get("attn_backend_cls"), mb.get("attn_backend_cls")
    if cls_a is not None or cls_b is not None:
        ctag = " [diff]" if cls_a != cls_b else " (match)"
        print(f"  attn_backend_cls: A={cls_a!r} B={cls_b!r}{ctag}")
    mdiff = _meta_line_diff(ma, mb)
    if mdiff:
        print("  meta diff (full recursive; branch keys also summarized above):")
        for line in mdiff:
            print(line)

    probes_a = da.get("probes") or {}
    probes_b = db.get("probes") or {}
    if not isinstance(probes_a, dict) or not isinstance(probes_b, dict):
        print("  probes: not dict  [diff]")
        return 1

    all_names = sorted(set(probes_a) | set(probes_b))
    for pname in all_names:
        binc, plines = _compare_one_probe(
            pname,
            probes_a.get(pname),
            probes_b.get(pname),
            strict=strict,
            cos_tol=cos_tol,
            compact=compact,
        )
        bad += binc
        for line in plines:
            print(line)

    fa, fb = da.get("full_tensors"), db.get("full_tensors")
    if fa is not None or fb is not None:
        ha = set(fa.keys()) if isinstance(fa, dict) else set()
        hb = set(fb.keys()) if isinstance(fb, dict) else set()
        if verbose or ha != hb:
            print(
                f"  full_tensors: A_keys={sorted(ha)} B_keys={sorted(hb)} "
                f"(binary compare skipped; inspect .pt manually)"
            )

    # Optional full tensors if future dumps add them
    for tk in ("tensor", "residual"):
        ta, tb = da.get(tk, None), db.get(tk, None)
        if ta is None and tb is None:
            continue
        if ta is None or tb is None:
            print(f"  {tk}: one is None  [diff]")
            bad += 1
            continue
        if isinstance(ta, torch.Tensor) and isinstance(tb, torch.Tensor):
            ta2, tb2, notes = _best_row_match_by_metrics(ta, tb, name=tk)
            for n in notes:
                print(f"  {n}")
            line, m = _tensor_stats(ta2, tb2, tk)
            print(line)
            if strict and m is not None and _strict_fail(m, cos_tol):
                bad += 1
    if verbose:
        print()
    return bad


def _attninner_key_sort(k: InnerKey) -> Tuple:
    stage_order = {
        "attn_dispatch": 0,
        "MHA_qkv": 1,
        "mla_prep_after_latent_norms": 2,
        "mla_prep_before_q_nope_pe_split": 3,
        "mla_prep_after_qkv_split": 4,
        "mla_prep_after_w_kc": 5,
        "mla_prep_after_rope": 6,
        "mla_core_inputs": 7,
        "mla_core_attn_mqa_inputs": 8,
        "mla_core_after_attn_mqa_raw": 9,
        "MLA_post_mqa": 10,
        "attn_inout": 11,
    }
    return (k.pos if k.pos >= 0 else 10**9, k.layer, stage_order.get(k.stage, 50), k.tp_rank)


def _collect_attninner_compare_jobs(
    req_a: Dict[str, Dict[InnerKey, str]],
    req_b: Dict[str, Dict[InnerKey, str]],
    mapping: Dict[str, str],
    only_stages: Optional[Set[str]],
) -> Tuple[List[Tuple[InnerKey, str, str]], int, int]:
    """Returns (jobs, missing_b_count, empty_req_pairs). Each job: (key_on_a, path_a, path_b)."""

    jobs: List[Tuple[InnerKey, str, str]] = []
    missing_b = 0
    empty_req_pairs = 0
    for ra, rb in sorted(mapping.items()):
        ia, ib = req_a.get(ra, {}), req_b.get(rb, {})
        keys_a, keys_b = set(ia.keys()), set(ib.keys())
        common = sorted(keys_a & keys_b, key=_attninner_key_sort)
        if only_stages is not None:
            common = [k for k in common if k.stage in only_stages]

        tp_map: Dict[int, int] = {}
        if not common:
            tpa = sorted({k.tp_rank for k in keys_a})
            tpb = sorted({k.tp_rank for k in keys_b})
            if len(tpa) == 1 and len(tpb) == 1 and tpa[0] != tpb[0]:
                tp_map[tpa[0]] = tpb[0]
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
                common = sorted(common2, key=_attninner_key_sort)

        if not common:
            empty_req_pairs += 1
            continue

        for k in common:
            kb = InnerKey(
                layer=k.layer,
                stage=k.stage,
                tp_rank=tp_map.get(k.tp_rank, k.tp_rank),
                pos=k.pos,
                decode_body=k.decode_body,
            )
            pb_path = ib.get(kb)
            if pb_path is None:
                missing_b += 1
                continue
            jobs.append((k, ia[k], pb_path))
    return jobs, missing_b, empty_req_pairs


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="Run A normal_forward dir")
    p.add_argument("--b", required=True, help="Run B normal_forward dir")
    p.add_argument("-o", "--output", default="", help="Write report to file (tee if TTY)")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--cos-tol", type=float, default=1e-5)
    p.add_argument(
        "--only-stages",
        default="",
        help="Comma-separated attninner stages; empty = all",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Full per-probe lines, full paths, legend (default is compact output).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar on stderr.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    dir_a = os.path.abspath(os.path.expanduser(args.a))
    dir_b = os.path.abspath(os.path.expanduser(args.b))
    if not os.path.isdir(dir_a) or not os.path.isdir(dir_b):
        print(f"ERROR: directory missing A={dir_a} B={dir_b}", file=sys.stderr)
        return 2

    only_stages: Optional[Set[str]] = None
    if (args.only_stages or "").strip():
        only_stages = {s.strip() for s in args.only_stages.split(",") if s.strip()}

    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else ""

    def run_body() -> int:
        req_a = _build_req_index(dir_a)
        req_b = _build_req_index(dir_b)
        mapping, notes = _build_req_mapping_attn(req_a, req_b)

        jobs, missing_b, empty_req_pairs = _collect_attninner_compare_jobs(
            req_a, req_b, mapping, only_stages
        )

        print(f"A: {dir_a}")
        print(f"B: {dir_b}")
        print(f"A attninner decode files: {len(_iter_attninner_decode_files(dir_a))}")
        print(f"B attninner decode files: {len(_iter_attninner_decode_files(dir_b))}")
        for n in notes:
            print(f"# {n}")
        print(f"mapped reqs: {len(mapping)}")
        print(f"compare jobs (slices): {len(jobs)}")
        if missing_b:
            print(f"# keys missing path on B after tp remap: {missing_b}")
        if empty_req_pairs:
            print(f"# req pairs with no common attninner keys: {empty_req_pairs}")
        if not mapping:
            print("\n# No request mapping — check attninner_*_decode_*.pt overlap.")
        print()

        if args.verbose:
            print(
                "  # stats.max_abs/mean/std (verbose): 各自 run 的抽样统计量，delta 小不是逐元 A−B；"
                " head/tail 为抽样序列 pairwise A vs B。"
            )
            print()

        use_tqdm = (
            tqdm is not None
            and not args.no_progress
            and sys.stderr.isatty()
            and len(jobs) > 0
        )
        if tqdm is None and not args.no_progress and sys.stderr.isatty():
            print(
                "(note: tqdm not installed; no progress bar — `pip install tqdm`)",
                file=sys.stderr,
            )

        bad = missing_b + empty_req_pairs
        compared = 0
        it: Any = jobs
        if use_tqdm:
            it = tqdm(
                jobs,
                desc="attninner",
                unit="slice",
                file=sys.stderr,
                mininterval=0.2,
            )
        for k, pa, pb in it:
            bad += compare_pair(
                k,
                pa,
                pb,
                strict=args.strict,
                cos_tol=args.cos_tol,
                verbose=args.verbose,
            )
            compared += 1

        print("--- summary ---")
        print(f"req pairs: {len(mapping)}")
        print(f"compared attninner keys: {compared}")
        print(f"bad score: {bad}")
        return 1 if args.strict and bad > 0 else 0

    if out_path:
        with open(out_path, "w", encoding="utf-8") as out_f:
            ctx: Any
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
