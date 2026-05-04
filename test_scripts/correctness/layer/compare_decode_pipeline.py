#!/usr/bin/env python3
"""
**Unified decode diff** for DP-attn vs special-dp-attn (or any two ``normal_forward/`` runs):
walk the forward pipeline in **semantic order** so you can see where drift accumulates.

Order per ``(decode_pos, decoder_layer, tp_rank on A)``:

1. ``layer_*`` **pre_attn**
2. ``attninner_*`` checkpoints (if present): ``attn_dispatch`` → ``MHA_qkv`` → ``MLA_post_mqa`` → ``attn_inout``
3. ``layer_*`` **post_attn** → **pre_mlp**
4. ``mlpinner_*`` inner stages (only files that exist): dense path then MoE path
5. ``layer_*`` **post_mlp** → optional **layer_out**

This merges ``compare_decode_layer_hiddens.py`` (``layer_*``),
``compare_decode_attn_inner.py`` (``attninner_*`` probes), and ``compare_decode_mlp_inner.py``
(``mlpinner_*``).

``compare_tensor.py`` stays for **prefill path** experiments (e.g. ``normal_forward`` vs
``split_prefill``); it is not decode req-mapped and is unchanged here.

Usage::

  python3 compare_decode_pipeline.py \
    --a run_0503_decode_dpattn_mlp_attn/normal_forward \
    --b run_0503_decode_special_dpattn_mlp_attn/normal_forward \
    -o run_0503_decode_dpattn_mlp_attn_vs_special_dpattn_mlp_attn.pipeline.diff

Options::

  --no-mlp-inner     Skip all ``mlpinner_*`` phases.
  --no-attn-inner    Skip all ``attninner_*`` phases.
  --no-aux           Skip pre_attn / post_attn (and any attninner sandwiched between them).
  --skip-layer-out   Do not emit layer_out at the end of each slice.
  --tee              With ``-o``, also copy the full report to stdout (slower on TTY).
  --no-progress      Disable the tqdm progress bar (stderr) when using ``-o``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, TextIO, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

# Allow ``python3 /path/to/compare_decode_pipeline.py`` from any cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import compare_decode_attn_inner as cai  # noqa: E402
import compare_decode_layer_hiddens as cdl  # noqa: E402
import compare_decode_mlp_inner as cmi  # noqa: E402


ATTNINNER_PIPELINE_STAGES: Tuple[str, ...] = (
    "attn_dispatch",
    "MHA_qkv",
    "mla_prep_after_latent_norms",
    "mla_prep_before_q_nope_pe_split",
    "mla_prep_after_qkv_split",
    "mla_prep_after_w_kc",
    "mla_prep_after_rope",
    "mla_core_inputs",
    "mla_core_after_attn_mqa_raw",
    "MLA_post_mqa",
    "attn_inout",
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


def _tp_map_for_layer_pair(
    ia: Dict[cdl.Key, str], ib: Dict[cdl.Key, str]
) -> Dict[int, int]:
    """Same idea as ``compare_decode_layer_hiddens`` when the key intersection is empty."""

    keys_a, keys_b = set(ia.keys()), set(ib.keys())
    common = keys_a & keys_b
    tp_map: Dict[int, int] = {}
    if not common:
        tpa = sorted({k.tp_rank for k in keys_a})
        tpb = sorted({k.tp_rank for k in keys_b})
        if len(tpa) == 1 and len(tpb) == 1 and tpa[0] != tpb[0]:
            tp_map[tpa[0]] = tpb[0]
    return tp_map


def _collect_layer_triples(ia: Dict[cdl.Key, str]) -> Set[Tuple[int, int, int]]:
    triples: Set[Tuple[int, int, int]] = set()
    for k in ia.keys():
        triples.add((k.pos, k.layer, k.tp_rank))
    return triples


def _collect_inner_triples(iia: Dict[cmi.InnerKey, str]) -> Set[Tuple[int, int, int]]:
    """Fallback when layer index is sparse but mlpinner dumps exist."""

    triples: Set[Tuple[int, int, int]] = set()
    for k in iia.keys():
        if k.pos >= 0:
            triples.add((k.pos, k.layer, k.tp_rank))
    return triples


def _build_pipeline_phases(
    *,
    include_aux: bool,
    include_attn_inner: bool,
    include_mlp_inner: bool,
    include_layer_out: bool,
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if include_aux:
        out.append(("layer", "pre_attn"))
    if include_attn_inner:
        for st in ATTNINNER_PIPELINE_STAGES:
            out.append(("attninner", st))
    if include_aux:
        out.append(("layer", "post_attn"))
    out.append(("layer", "pre_mlp"))
    if include_mlp_inner:
        out.extend(
            [
                ("mlpinner", "dense_gate_up"),
                ("mlpinner", "dense_act"),
                ("mlpinner", "dense_down_proj"),
                ("mlpinner", "moe_router_meta"),
                ("mlpinner", "moe_routed_maybe_scaled_pre_shared"),
                ("mlpinner", "moe_pre_tp_allreduce"),
            ]
        )
    out.append(("layer", "post_mlp"))
    if include_layer_out:
        out.append(("layer", "layer_out"))
    return out


def _decode_body_for_pos(pos: int) -> str:
    return f"pos{int(pos)}"


def _find_inner_path(
    req_index: Dict[str, Dict[cmi.InnerKey, str]],
    *,
    prefer_req: str,
    layer: int,
    stage: str,
    tp: int,
    pos: int,
) -> Optional[str]:
    """Resolve path when ``mlpinner_*`` req id in filename differs from ``layer_*`` for the same run."""

    dbody = _decode_body_for_pos(pos)
    want = cmi.InnerKey(
        layer=layer, stage=stage, tp_rank=tp, pos=pos, decode_body=dbody
    )
    pr = req_index.get(prefer_req) or {}
    if want in pr:
        return pr[want]
    for _rid, idx in sorted(req_index.items()):
        if want in idx:
            return idx[want]
    return None


def _find_attninner_path(
    req_index: Dict[str, Dict[cmi.InnerKey, str]],
    *,
    prefer_req: str,
    layer: int,
    stage: str,
    tp: int,
    pos: int,
) -> Optional[str]:
    """Resolve ``attninner_*`` when req id differs from ``layer_*`` (same fallback as mlpinner)."""

    dbody = _decode_body_for_pos(pos)
    want = cmi.InnerKey(
        layer=layer, stage=stage, tp_rank=tp, pos=pos, decode_body=dbody
    )
    pr = req_index.get(prefer_req) or {}
    if want in pr:
        return pr[want]
    for _rid, idx in sorted(req_index.items()):
        if want in idx:
            return idx[want]
    return None


def _triples_for_req_on_a(
    ra: str,
    *,
    req_layer_a: Dict[str, Dict[cdl.Key, str]],
    req_inner_a: Dict[str, Dict[cmi.InnerKey, str]],
    req_attn_a: Dict[str, Dict[cmi.InnerKey, str]],
) -> List[Tuple[int, int, int]]:
    ia = req_layer_a.get(ra, {})
    iia = req_inner_a.get(ra, {})
    atta = req_attn_a.get(ra, {})
    return sorted(
        _collect_layer_triples(ia)
        | _collect_inner_triples(iia)
        | _collect_inner_triples(atta)
    )


def _count_phase_steps(
    mapping: Dict[str, str],
    *,
    req_layer_a: Dict[str, Dict[cdl.Key, str]],
    req_inner_a: Dict[str, Dict[cmi.InnerKey, str]],
    req_attn_a: Dict[str, Dict[cmi.InnerKey, str]],
    phases: List[Tuple[str, str]],
) -> int:
    total = 0
    for ra, _rb in sorted(mapping.items()):
        triples = _triples_for_req_on_a(
            ra,
            req_layer_a=req_layer_a,
            req_inner_a=req_inner_a,
            req_attn_a=req_attn_a,
        )
        total += len(triples) * len(phases)
    return total


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="Run A normal_forward dir")
    p.add_argument("--b", required=True, help="Run B normal_forward dir")
    p.add_argument(
        "-o",
        "--output",
        default="",
        help="Write full report to this file; terminal shows a short banner + tqdm (stderr) by default.",
    )
    p.add_argument(
        "--tee",
        action="store_true",
        help="With -o, also stream the full report to stdout (can be very slow on a TTY).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not show tqdm when -o is set.",
    )
    p.add_argument("--strict", action="store_true", help="Non-zero exit if any tensor fails cos tolerance")
    p.add_argument(
        "--strict-primary",
        action="store_true",
        help="Non-zero exit only if pre_mlp tensors fail cos tolerance",
    )
    p.add_argument("--cos-tol", type=float, default=1e-5)
    p.add_argument(
        "--no-mlp-inner",
        action="store_true",
        help="Skip mlpinner_* phases (only layer stages, still pipeline-ordered).",
    )
    p.add_argument(
        "--no-attn-inner",
        action="store_true",
        help="Skip attninner_* probes (attention inner checkpoints).",
    )
    p.add_argument(
        "--no-aux",
        action="store_true",
        help="Skip pre_attn / attninner / post_attn (start at pre_mlp).",
    )
    p.add_argument(
        "--skip-layer-out",
        action="store_true",
        help="Do not compare layer_out at end of each (pos, layer, tp) slice.",
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

    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else ""

    phases = _build_pipeline_phases(
        include_aux=not args.no_aux,
        include_attn_inner=not args.no_aux and not args.no_attn_inner,
        include_mlp_inner=not args.no_mlp_inner,
        include_layer_out=not args.skip_layer_out,
    )

    report_to_file_only = bool(out_path) and not (args.tee and sys.stdout.isatty())

    def run_body(
        *,
        stderr_progress: bool,
        echo_summary_stderr: bool,
    ) -> int:
        req_layer_a = cdl._build_req_index(dir_a)
        req_layer_b = cdl._build_req_index(dir_b)
        mapping, notes = cdl._build_req_mapping(req_layer_a, req_layer_b)

        req_inner_a = cmi._build_req_index(dir_a)
        req_inner_b = cmi._build_req_index(dir_b)
        req_attn_a = cai._build_req_index(dir_a)
        req_attn_b = cai._build_req_index(dir_b)

        if not mapping:
            mapping, notes2 = cmi._build_req_mapping(req_inner_a, req_inner_b)
            notes.extend(notes2)
        if not mapping:
            mapping3, notes3 = cai._build_req_mapping_attn(req_attn_a, req_attn_b)
            mapping.update(mapping3)
            notes.extend(notes3)

        step_total = _count_phase_steps(
            mapping,
            req_layer_a=req_layer_a,
            req_inner_a=req_inner_a,
            req_attn_a=req_attn_a,
            phases=phases,
        )
        prog: Any = None
        if stderr_progress and tqdm is not None and sys.stderr.isatty():
            prog = tqdm(
                total=max(step_total, 1),
                file=sys.stderr,
                unit="step",
                desc="pipeline",
                mininterval=0.2,
            )

        print(f"A: {dir_a}")
        print(f"B: {dir_b}")
        print(f"A layer decode files: {len(cdl._iter_decode_files(dir_a))}")
        print(f"B layer decode files: {len(cdl._iter_decode_files(dir_b))}")
        print(f"A attninner decode files: {len(cai._iter_attninner_decode_files(dir_a))}")
        print(f"B attninner decode files: {len(cai._iter_attninner_decode_files(dir_b))}")
        print(f"A mlpinner decode files: {len(cmi._iter_mlpinner_decode_files(dir_a))}")
        print(f"B mlpinner decode files: {len(cmi._iter_mlpinner_decode_files(dir_b))}")
        print(f"A req groups (layer): {len(req_layer_a)}")
        print(f"B req groups (layer): {len(req_layer_b)}")
        print(f"mapped reqs: {len(mapping)}")
        for n in notes:
            print(f"# {n}")
        print()
        print(
            "# Pipeline order per (pos, layer, tp_A): "
            + " → ".join(f"{k}/{s}" for k, s in phases)
        )
        print()

        bad = 0
        bad_primary = 0
        compared = 0
        compared_primary = 0

        try:
            for ra, rb in sorted(mapping.items()):
                ia = req_layer_a.get(ra, {})
                ib = req_layer_b.get(rb, {})
                iia = req_inner_a.get(ra, {})
                iib = req_inner_b.get(rb, {})
                atta = req_attn_a.get(ra, {})
                attb = req_attn_b.get(rb, {})

                tp_map = _tp_map_for_layer_pair(ia, ib)
                if tp_map:
                    print(
                        f"== req pair A={ra} B={rb}: layer tp_rank remap "
                        f"A{list(tp_map.keys())[0]}->B{list(tp_map.values())[0]} =="
                    )

                triples = _triples_for_req_on_a(
                    ra,
                    req_layer_a=req_layer_a,
                    req_inner_a=req_inner_a,
                    req_attn_a=req_attn_a,
                )
                if not triples:
                    print(
                        f"== req pair A={ra} B={rb}: no decode keys on A "
                        f"(layer / mlpinner / attninner); skip ==="
                    )
                    bad += 1
                    continue

                print(f"== req pair A={ra} B={rb}: pipeline slices={len(triples)} ===")

                for pos, layer, tp_a in triples:
                    print()
                    print(
                        f"### pipeline slice  pos={pos}  layer={layer}  tp_A={tp_a}  "
                        f"(req A={ra[:12]}…  B={rb[:12]}…) ###"
                    )
                    tp_b = tp_map.get(tp_a, tp_a)

                    for step_i, (kind, stg) in enumerate(phases, start=1):
                        tag = f"[{step_i}/{len(phases)}] {kind}:{stg}"
                        if kind == "layer":
                            ka = cdl.Key(
                                layer=layer, stage=stg, tp_rank=tp_a, pos=pos
                            )
                            kb = cdl.Key(
                                layer=layer, stage=stg, tp_rank=tp_b, pos=pos
                            )
                            if ka not in ia or kb not in ib:
                                print(
                                    f"--- {tag}  SKIP "
                                    f"(missing A={ka in ia} B={kb in ib})"
                                )
                                if prog is not None:
                                    prog.update(1)
                                continue
                            print(f">>> {tag}")
                            use_strict = args.strict or (
                                args.strict_primary and stg == "pre_mlp"
                            )
                            b = cdl.compare_pair(
                                ka,
                                ia[ka],
                                ib[kb],
                                strict=use_strict,
                                cos_tol=args.cos_tol,
                            )
                            bad += b
                            compared += 1
                            if stg == "pre_mlp":
                                compared_primary += 1
                                bad_primary += b
                        elif kind == "attninner":
                            pa_i = _find_attninner_path(
                                req_attn_a,
                                prefer_req=ra,
                                layer=layer,
                                stage=stg,
                                tp=tp_a,
                                pos=pos,
                            )
                            pb_i = _find_attninner_path(
                                req_attn_b,
                                prefer_req=rb,
                                layer=layer,
                                stage=stg,
                                tp=tp_b,
                                pos=pos,
                            )
                            if pa_i is None or pb_i is None:
                                if prog is not None:
                                    prog.update(1)
                                continue
                            print(f">>> {tag}")
                            b = cai.compare_pair(
                                cmi.InnerKey(
                                    layer=layer,
                                    stage=stg,
                                    tp_rank=tp_a,
                                    pos=pos,
                                    decode_body=_decode_body_for_pos(pos),
                                ),
                                pa_i,
                                pb_i,
                                strict=args.strict,
                                cos_tol=args.cos_tol,
                            )
                            bad += b
                            compared += 1
                        else:  # mlpinner
                            dbody = _decode_body_for_pos(pos)
                            ika = cmi.InnerKey(
                                layer=layer,
                                stage=stg,
                                tp_rank=tp_a,
                                pos=pos,
                                decode_body=dbody,
                            )
                            pa_i = _find_inner_path(
                                req_inner_a,
                                prefer_req=ra,
                                layer=layer,
                                stage=stg,
                                tp=tp_a,
                                pos=pos,
                            )
                            pb_i = _find_inner_path(
                                req_inner_b,
                                prefer_req=rb,
                                layer=layer,
                                stage=stg,
                                tp=tp_b,
                                pos=pos,
                            )
                            if pa_i is None or pb_i is None:
                                # Dense vs MoE: quiet skip.
                                if prog is not None:
                                    prog.update(1)
                                continue
                            print(f">>> {tag}")
                            b = cmi.compare_pair(
                                ika,
                                pa_i,
                                pb_i,
                                strict=args.strict,
                                cos_tol=args.cos_tol,
                            )
                            bad += b
                            compared += 1
                        if prog is not None:
                            prog.update(1)

        finally:
            if prog is not None:
                prog.close()

        print("--- summary ---")
        print(f"req pairs: {len(mapping)}")
        print(f"compared pair-calls (layer + attninner + mlpinner): {compared}")
        print(f"  of which pre_mlp (primary) calls: {compared_primary}")
        print(f"bad score (all): {bad}")
        print(f"bad score (pre_mlp only): {bad_primary}")
        if echo_summary_stderr:
            print("", file=sys.stderr)
            print(
                f"compare_decode_pipeline: done — compared={compared} bad={bad} "
                f"bad_primary={bad_primary}",
                file=sys.stderr,
            )
        if args.strict and bad > 0:
            return 1
        if args.strict_primary and bad_primary > 0:
            return 1
        return 0

    if out_path:
        if report_to_file_only:
            print(
                "compare_decode_pipeline: full report → "
                f"{out_path}\n"
                f"A: {dir_a}\n"
                f"B: {dir_b}",
                file=sys.stderr,
            )
            if tqdm is None and not args.no_progress:
                print(
                    "(note: tqdm not installed; progress bar disabled — `pip install tqdm`)",
                    file=sys.stderr,
                )
        stderr_progress = (
            tqdm is not None
            and not args.no_progress
            and sys.stderr.isatty()
            and report_to_file_only
        )
        with open(out_path, "w", encoding="utf-8") as out_f:
            redirect_tgt: Any = (
                _TeeStdout(sys.stdout, out_f)
                if args.tee and sys.stdout.isatty()
                else out_f
            )
            with contextlib.redirect_stdout(redirect_tgt):
                return run_body(
                    stderr_progress=stderr_progress,
                    echo_summary_stderr=report_to_file_only,
                )

    return run_body(stderr_progress=False, echo_summary_stderr=False)


if __name__ == "__main__":
    raise SystemExit(main())
