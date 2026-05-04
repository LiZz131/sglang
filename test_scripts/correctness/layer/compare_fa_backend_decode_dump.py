#!/usr/bin/env python3
"""Compare ``fa_backend_decode/*.pt`` across two runs with **decode signature** req pairing.

**v2 filenames** (``SGLANG_DEBUG_DUMP_FA_BACKEND`` with current sglang): mirror
``deepseek_v2._debug_decode_dump_filename_suffix``::

    fa_decode_layer_{L:04d}_tp{R}_decode_{pospart}_req{req}.pt

Pairing (same spirit as ``compare_decode_layer_hiddens.py``):

1. **Within each run**, group files by the **filename token after ``_req``** (opaque id from
   ``_debug_req_ids``); this is only a **local bucket id**, not used for equality across runs.
2. For each bucket compute **ReqSig** = (decode_min_pos, decode_max_pos, decode_pos_count) from
   the set of decode positions in its files — this is the **cross-run stable signature**.
3. Map **A_bucket → B_bucket** when **ReqSig** matches (sorted order if multiple buckets share a
   signature). **Never** requires the same HTTP / scheduler ``req_id`` string on A and B.
4. Within each mapped pair, optional **tp_rank remap** (singleton tp mismatch).
5. Output order: **req pair** (shown as A/B bucket ids) → **decode_pos** → **layer** → diffs.

**v1 legacy** (``fa_decode_layer_*_tp*_pos*.pt`` without ``_decode_``): all files are grouped under
synthetic req ``__legacy__`` and paired by ``(layer, tp, pos)`` only (no cross-run req signature).

Examples::

    python3 compare_fa_backend_decode_dump.py --preset 0504_special_vs_dp \\
      --ignore-forward-batch-scalars -o run_0504.fa_decode.diff

python3 compare_fa_backend_decode_dump.py \
    --a /sgl-workspace/sglang/test_scripts/correctness/layer/run_0504_decode_dpattn_fa/normal_forward/fa_backend_decode \
    --b /sgl-workspace/sglang/test_scripts/correctness/layer/run_0504_decode_special_dpattn_fa/normal_forward/fa_backend_decode \
    --ignore-forward-batch-scalars \
    -o run_0504_special_vs_dpattn.fa_decode.diff
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

_LAYER_DIR = os.path.dirname(os.path.abspath(__file__))

_PRESETS: Dict[str, Tuple[str, str]] = {
    "0504_special_vs_dp": (
        os.path.join(_LAYER_DIR, "run_0504_decode_special_dpattn_fa", "normal_forward", "fa_backend_decode"),
        os.path.join(_LAYER_DIR, "run_0504_decode_dpattn_fa", "normal_forward", "fa_backend_decode"),
    ),
}

# Legacy: fa_decode_layer_0000_tp0_pos10.pt
_FA_RE_V1 = re.compile(
    r"^fa_decode_layer_(?P<layer>\d+)_tp(?P<tp>\d+)_pos(?P<pos>-?\d+)\.pt$"
)


@dataclass(frozen=True)
class ReqSig:
    decode_min_pos: int
    decode_max_pos: int
    decode_pos_count: int


@dataclass
class ParsedFA:
    path: str
    layer: int
    tp: int
    req: str
    pos_sort: int
    pos_raw: str
    version: str  # "v2" | "v1"


def _primary_pos_sort(pos_raw: str) -> int:
    """Stable int for sorting / ReqSig; mirrors single-token ``pos{N}`` common case."""
    if pos_raw.startswith("pos") and len(pos_raw) > 3:
        tail = pos_raw[3:]
        if tail.lstrip("-").isdigit():
            return int(tail)
    if pos_raw == "nopos":
        return -(10**9)
    # multi-token / exotic: stable bucket
    return -(abs(hash(pos_raw)) % (10**9))


def parse_fa_filename(path: str) -> Optional[ParsedFA]:
    fn = os.path.basename(path)
    if not fn.endswith(".pt") or not fn.startswith("fa_decode_layer_"):
        return None
    m1 = _FA_RE_V1.match(fn)
    # v2 names use ``..._tp{R}_decode_pos..._req...`` and never match this regex.
    if m1:
        layer = int(m1.group("layer"))
        tp = int(m1.group("tp"))
        pos = int(m1.group("pos"))
        return ParsedFA(
            path=path,
            layer=layer,
            tp=tp,
            req="__legacy__",
            pos_sort=pos,
            pos_raw=f"pos{pos}",
            version="v1",
        )
    m2 = re.match(r"^fa_decode_layer_(\d+)_tp(\d+)_(.+)\.pt$", fn)
    if not m2:
        return None
    layer, tp, rest = int(m2.group(1)), int(m2.group(2)), m2.group(3)
    # On disk: ``tp0`` + ``_decode_pos10_req...`` → ``tp0_decode_pos10_req...`` (single ``_``
    # between rank and ``decode``). Same pattern as layer hidden dumps.
    pos_raw: str
    req: str
    if "_req" not in rest:
        return None
    ri = rest.rfind("_req")
    mid = rest[:ri]
    req = rest[ri + len("_req") :]
    if mid.startswith("_decode_"):
        pos_raw = mid[len("_decode_") :]
    elif mid.startswith("decode_"):
        pos_raw = mid[len("decode_") :]
    else:
        return None
    pos_sort = _primary_pos_sort(pos_raw)
    return ParsedFA(
        path=path,
        layer=layer,
        tp=tp,
        req=req,
        pos_sort=pos_sort,
        pos_raw=pos_raw,
        version="v2",
    )


def _iter_parsed(dir_path: str) -> List[ParsedFA]:
    out: List[ParsedFA] = []
    if not os.path.isdir(dir_path):
        return out
    for fn in sorted(os.listdir(dir_path)):
        if not fn.endswith(".pt"):
            continue
        p = os.path.join(dir_path, fn)
        parsed = parse_fa_filename(p)
        if parsed is not None:
            out.append(parsed)
    return out


def _req_sig_from_parsed(paths: Iterable[ParsedFA]) -> Optional[ReqSig]:
    poss: List[int] = []
    for p in paths:
        poss.append(p.pos_sort)
    if not poss:
        return None
    poss_u = sorted(set(poss))
    return ReqSig(
        decode_min_pos=poss_u[0],
        decode_max_pos=poss_u[-1],
        decode_pos_count=len(poss_u),
    )


# req -> (layer, pos_sort) -> tp_rank -> path
FAIndex = Dict[str, Dict[Tuple[int, int], Dict[int, str]]]


def _build_fa_index(dir_path: str) -> FAIndex:
    by_req: FAIndex = {}
    for p in _iter_parsed(dir_path):
        by_req.setdefault(p.req, {}).setdefault((p.layer, p.pos_sort), {})[p.tp] = p.path
    return by_req


def _build_req_mapping(
    reqs_a: FAIndex, reqs_b: FAIndex
) -> Tuple[Dict[str, str], List[str]]:
    notes: List[str] = []
    sig_to_a: Dict[ReqSig, List[str]] = {}
    sig_to_b: Dict[ReqSig, List[str]] = {}

    for rid, idx in reqs_a.items():
        paths = []
        for (_layer, _ps), d in idx.items():
            for tp, path in d.items():
                pr = parse_fa_filename(path)
                if pr:
                    paths.append(pr)
        sig = _req_sig_from_parsed(paths)
        if sig is not None:
            sig_to_a.setdefault(sig, []).append(rid)
    for rid, idx in reqs_b.items():
        paths = []
        for (_layer, _ps), d in idx.items():
            for _tp, path in d.items():
                pr = parse_fa_filename(path)
                if pr:
                    paths.append(pr)
        sig = _req_sig_from_parsed(paths)
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


def _reqsig_sets(idx: FAIndex) -> Tuple[Set[ReqSig], Dict[ReqSig, List[str]]]:
    """Unique ReqSig per run + which local bucket ids share each sig."""
    sig_to_rids: Dict[ReqSig, List[str]] = {}
    for rid, sub in idx.items():
        paths: List[ParsedFA] = []
        for _lp, d in sub.items():
            for _tp, path in d.items():
                pr = parse_fa_filename(path)
                if pr:
                    paths.append(pr)
        sig = _req_sig_from_parsed(paths)
        if sig is not None:
            sig_to_rids.setdefault(sig, []).append(rid)
    return set(sig_to_rids), sig_to_rids


def _infer_tp_map(
    idx_a: Dict[Tuple[int, int], Dict[int, str]],
    idx_b: Dict[Tuple[int, int], Dict[int, str]],
) -> Tuple[Dict[int, int], List[str]]:
    """If each (layer,pos) has a single tp on each side and tp sets disagree consistently, remap."""
    notes: List[str] = []
    pairs: List[Tuple[int, int]] = []
    keys_a = set(idx_a.keys())
    keys_b = set(idx_b.keys())
    for lp in sorted(keys_a & keys_b):
        ta = sorted(idx_a[lp].keys())
        tb = sorted(idx_b[lp].keys())
        if len(ta) == 1 and len(tb) == 1 and ta[0] != tb[0]:
            pairs.append((ta[0], tb[0]))
    if not pairs:
        return {}, notes
    first = pairs[0]
    if all(p == first for p in pairs):
        return {first[0]: first[1]}, notes
    notes.append(
        f"tp remap ambiguous across slices (sample pairs {pairs[:8]}); using identity tp map"
    )
    return {}, notes


def _load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_layers(s: str) -> Optional[Set[int]]:
    s = (s or "").strip()
    if not s:
        return None
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a64 = a.detach().float().flatten()
    b64 = b.detach().float().flatten()
    if a64.shape != b64.shape:
        return {"max_abs": float("nan"), "cos": float("nan")}
    d = (a64 - b64).abs()
    max_abs = float(d.max().item()) if d.numel() else 0.0
    na = float(torch.linalg.norm(a64))
    nb = float(torch.linalg.norm(b64))
    if na < 1e-30 or nb < 1e-30:
        cos = 1.0 if max_abs < 1e-30 else 0.0
    else:
        cos = float((a64 @ b64) / na / nb)
    return {"max_abs": max_abs, "cos": cos}


def _tensor_diff(
    ta: torch.Tensor, tb: torch.Tensor, max_abs_tol: float, cos_tol: float
) -> bool:
    if ta.shape != tb.shape:
        return True
    m = _tensor_metrics(ta, tb)
    if not math.isfinite(m["max_abs"]) or not math.isfinite(m["cos"]):
        return True
    if m["max_abs"] > max_abs_tol:
        return True
    if abs(1.0 - m["cos"]) > cos_tol:
        return True
    return False


def _fmt_tensor_diff(ta: torch.Tensor, tb: torch.Tensor) -> str:
    m = _tensor_metrics(ta, tb)
    return f"max_abs={m['max_abs']:.10e} cos={m['cos']:.12f}"


def _compare_dict_tensors(
    da: Dict[str, Any],
    db: Dict[str, Any],
    prefix: str,
    *,
    max_abs_tol: float,
    cos_tol: float,
) -> List[str]:
    lines: List[str] = []
    ka = set(da) if isinstance(da, dict) else set()
    kb = set(db) if isinstance(db, dict) else set()
    for k in sorted(ka - kb):
        lines.append(f"    {prefix} only A: {k!r}")
    for k in sorted(kb - ka):
        lines.append(f"    {prefix} only B: {k!r}")
    for k in sorted(ka & kb):
        ta, tb = da.get(k), db.get(k)
        if not torch.is_tensor(ta) or not torch.is_tensor(tb):
            if ta != tb:
                lines.append(f"    {prefix}[{k!r}] non-tensor diff A={ta!r} B={tb!r}")
            continue
        if ta.shape != tb.shape:
            lines.append(
                f"    {prefix}[{k!r}] shape A={tuple(ta.shape)} B={tuple(tb.shape)}"
            )
            continue
        if _tensor_diff(ta, tb, max_abs_tol, cos_tol):
            lines.append(f"    {prefix}[{k!r}] {_fmt_tensor_diff(ta, tb)}")
    return lines


def _compare_payloads(
    pa: str,
    pb: str,
    *,
    ignore_dbg_flags: bool,
    ignore_forward_batch_scalars: bool,
    sections: Set[str],
    max_abs_tol: float,
    cos_tol: float,
) -> Tuple[int, List[str]]:
    da, db = _load(pa), _load(pb)
    lines: List[str] = []
    if not ignore_dbg_flags and da.get("dbg_flags") != db.get("dbg_flags"):
        lines.append(f"    dbg_flags A={da.get('dbg_flags')!r}")
        lines.append(f"             B={db.get('dbg_flags')!r}")
    if not ignore_forward_batch_scalars and da.get("forward_batch_scalars") != db.get(
        "forward_batch_scalars"
    ):
        lines.append(
            "    forward_batch_scalars differ (use --ignore-forward-batch-scalars to skip)"
        )
        sa, sb = da.get("forward_batch_scalars"), db.get("forward_batch_scalars")
        if isinstance(sa, dict) and isinstance(sb, dict):
            lines.append(
                f"      keys A-only={sorted(set(sa) - set(sb))[:12]} "
                f"B-only={sorted(set(sb) - set(sa))[:12]}"
            )

    for sec in (
        "forward_batch_tensors",
        "flash_forward_metadata",
        "req_to_token_rows",
        "kv_buffers",
    ):
        if sec not in sections:
            continue
        sa, sb = da.get(sec), db.get(sec)
        if type(sa) != type(sb):
            lines.append(f"    {sec}: type A={type(sa)} B={type(sb)}")
            continue
        if isinstance(sa, dict) and isinstance(sb, dict):
            lines.extend(
                _compare_dict_tensors(
                    sa, sb, sec, max_abs_tol=max_abs_tol, cos_tol=cos_tol
                )
            )

    for sec in (
        "inputs_q",
        "inputs_k",
        "inputs_v",
        "inputs_q_rope",
        "inputs_k_rope",
        "out_o_before_view",
    ):
        if sec not in sections:
            continue
        ta, tb = da.get(sec), db.get(sec)
        if ta is None and tb is None:
            continue
        if not torch.is_tensor(ta) or not torch.is_tensor(tb):
            lines.append(f"    {sec}: missing or non-tensor A={type(ta)} B={type(tb)}")
            continue
        if ta.shape != tb.shape:
            lines.append(
                f"    {sec}: shape A={tuple(ta.shape)} B={tuple(tb.shape)}"
            )
            continue
        if _tensor_diff(ta, tb, max_abs_tol, cos_tol):
            lines.append(f"    {sec}: {_fmt_tensor_diff(ta, tb)}")

    detail = [f"      A {pa}", f"      B {pb}"]
    if lines:
        return len(lines), detail + lines
    return 0, []


def _default_sections() -> Set[str]:
    return {
        "forward_batch_tensors",
        "flash_forward_metadata",
        "req_to_token_rows",
        "kv_buffers",
        "inputs_q",
        "inputs_k",
        "inputs_v",
        "inputs_q_rope",
        "inputs_k_rope",
        "out_o_before_view",
    }


def _parse_sections(s: str) -> Set[str]:
    s = (s or "").strip().lower()
    if not s or s == "all":
        return _default_sections()
    allowed = _default_sections()
    out: Set[str] = set()
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x not in allowed:
            raise SystemExit(f"unknown section {x!r}; allowed: {sorted(allowed)}")
        out.add(x)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare FA backend decode dumps with decode-signature req pairing + tp remap."
    )
    p.add_argument("--a", default="", help="Run A: .../fa_backend_decode")
    p.add_argument("--b", default="", help="Run B: .../fa_backend_decode")
    p.add_argument(
        "--preset",
        default="",
        choices=list(_PRESETS.keys()),
        help="Builtin --a/--b paths",
    )
    p.add_argument("-o", "--output", default="", help="Write report file")
    p.add_argument("--tee", action="store_true", help="With -o, also print full report")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="With -o: only summary on stderr (full report still in file)",
    )
    p.add_argument("--only-layers", default="", help="Comma layer ids")
    p.add_argument("--ignore-dbg-flags", action="store_true")
    p.add_argument("--ignore-forward-batch-scalars", action="store_true")
    p.add_argument("--sections", default="all")
    p.add_argument("--max-abs-tol", type=float, default=0.0)
    p.add_argument("--cos-tol", type=float, default=1e-5)
    p.add_argument("--fail-on-diff", action="store_true")
    args = p.parse_args()

    dir_a = (args.a or "").strip()
    dir_b = (args.b or "").strip()
    if args.preset:
        pa, pb = _PRESETS[args.preset]
        if not dir_a:
            dir_a = pa
        if not dir_b:
            dir_b = pb
    if not dir_a or not dir_b:
        p.error("need --a and --b or --preset")

    dir_a = os.path.abspath(os.path.expanduser(dir_a))
    dir_b = os.path.abspath(os.path.expanduser(dir_b))
    only_layers = _parse_layers(args.only_layers)
    sections = _parse_sections(args.sections)

    idx_a = _build_fa_index(dir_a)
    idx_b = _build_fa_index(dir_b)
    mapping, map_notes = _build_req_mapping(idx_a, idx_b)

    all_print: List[str] = [
        f"A: {dir_a}  parsed_files={sum(len(d) for r in idx_a.values() for d in r.values())}",
        f"B: {dir_b}  parsed_files={sum(len(d) for r in idx_b.values() for d in r.values())}",
        f"req_mapping count={len(mapping)}",
    ]
    for n in map_notes:
        all_print.append(f"NOTE: {n}")

    if not mapping and idx_a and idx_b:
        # Legacy-only: single bucket
        if set(idx_a.keys()) == {"__legacy__"} and set(idx_b.keys()) == {"__legacy__"}:
            mapping = {"__legacy__": "__legacy__"}
            all_print.append("NOTE: legacy v1 filenames only; using synthetic req __legacy__")
        else:
            sa, _ma = _reqsig_sets(idx_a)
            sb, _mb = _reqsig_sets(idx_b)
            if sa or sb:
                only_a = sorted(sa - sb, key=lambda s: (s.decode_min_pos, s.decode_max_pos))
                only_b = sorted(sb - sa, key=lambda s: (s.decode_min_pos, s.decode_max_pos))
                all_print.append(
                    "NOTE: no common ReqSig between A and B (decode pos coverage differs?). "
                    "Pairing uses **ReqSig** = (min_pos, max_pos, num_distinct_decode_positions), "
                    "not raw req_id string equality across runs."
                )
                if only_a:
                    all_print.append(f"  ReqSig only on A (sample): {only_a[:6]}")
                if only_b:
                    all_print.append(f"  ReqSig only on B (sample): {only_b[:6]}")

    total_diff_lines = 0
    slices_with_diff = 0
    missing_pairs = 0

    for ra, rb in sorted(mapping.items(), key=lambda kv: (kv[0], kv[1])):
        sub_a = idx_a.get(ra, {})
        sub_b = idx_b.get(rb, {})
        tp_map, tp_notes = _infer_tp_map(sub_a, sub_b)
        all_print.append("")
        all_print.append(f"######## req pair A_req={ra!r}  B_req={rb!r} ########")
        for tn in tp_notes:
            all_print.append(f"  NOTE: {tn}")
        if tp_map:
            all_print.append(f"  tp_rank remap (A->B): {tp_map}")

        keys_a = set(sub_a.keys())
        keys_b = set(sub_b.keys())
        common_lp = sorted(keys_a & keys_b, key=lambda x: (x[1], x[0]))
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        if only_a:
            all_print.append(
                f"  only A (layer,pos_sort): {only_a[:24]}{' ...' if len(only_a) > 24 else ''}"
            )
            missing_pairs += len(only_a)
        if only_b:
            all_print.append(
                f"  only B (layer,pos_sort): {only_b[:24]}{' ...' if len(only_b) > 24 else ''}"
            )
            missing_pairs += len(only_b)

        for layer, pos_s in common_lp:
            if only_layers is not None and layer not in only_layers:
                continue
            row_a = sub_a.get((layer, pos_s), {})
            row_b = sub_b.get((layer, pos_s), {})
            pos_raw = ""
            for tp_a, path_a in sorted(row_a.items()):
                tp_b = tp_map.get(tp_a, tp_a)
                path_b = row_b.get(tp_b)
                if path_b is None:
                    if len(row_b) == 1:
                        only_tb = next(iter(row_b.keys()))
                        path_b = row_b[only_tb]
                        tp_b = only_tb
                    else:
                        all_print.append(
                            f"  --- layer={layer} pos_sort={pos_s} tp_A={tp_a} -> missing B tp={tp_b} ---"
                        )
                        missing_pairs += 1
                        continue
                pr = parse_fa_filename(path_a)
                if pr:
                    pos_raw = pr.pos_raw
                n, chunk = _compare_payloads(
                    path_a,
                    path_b,
                    ignore_dbg_flags=bool(args.ignore_dbg_flags),
                    ignore_forward_batch_scalars=bool(
                        args.ignore_forward_batch_scalars
                    ),
                    sections=sections,
                    max_abs_tol=float(args.max_abs_tol),
                    cos_tol=float(args.cos_tol),
                )
                all_print.append(
                    f"  --- decode_pos={pos_raw or pos_s}  layer={layer}  "
                    f"tp_A={tp_a}  tp_B={tp_b} ---"
                )
                if n > 0:
                    slices_with_diff += 1
                    total_diff_lines += n
                    all_print.extend(chunk)
                else:
                    all_print.append("    (match within tolerances)")

    summary = (
        f"\n--- summary: req_pairs={len(mapping)} slices_with_diff={slices_with_diff} "
        f"diff_lines={total_diff_lines} missing_lp={missing_pairs} ---"
    )
    all_print.append(summary)

    text = "\n".join(all_print) + "\n"
    out_path = (args.output or "").strip()
    if out_path:
        ap = os.path.abspath(out_path)
        with open(ap, "w", encoding="utf-8") as wf:
            wf.write(text)
        print(f"Wrote report: {ap}", file=sys.stderr)
        if args.tee:
            print(text, end="")
        elif args.quiet:
            print(summary.strip(), file=sys.stderr)
        else:
            print(summary, file=sys.stderr)
    elif not args.quiet:
        print(text, end="")
    else:
        print(summary.strip(), file=sys.stderr)

    if args.fail_on_diff and (slices_with_diff or missing_pairs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
