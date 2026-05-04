#!/usr/bin/env python3
"""
Align KV-cache fingerprint dumps with **greedy sampling** alignment, then report
whether KV differs when top-1 sampled token **still matches** between two runs.

This answers: “在未发生 token 跳变时，KV cache 取样是否仍存在数值误差 / 漂移？”

Typical inputs (same layout as offline dumps):
  <run>/kvcache_fingerprint/kvcache_layer0000_attn_prepare_tp{r}_pos{pos}_req{rid}.pt
  <run>/<mode_tag>/<rid>/decode/pos_{pos}/tp{r}_raw.pt

Usage example (0501 three runs):

  cd test_scripts/correctness/layer
  python3 analyze_kvcache_token_aligned.py \\
    --baseline run_0501_decode_tp,tp \\
    --other   run_0501_decode_dp_attn,dp_attention

  python3 analyze_kvcache_token_aligned.py \\
    --baseline run_0501_decode_tp,tp \\
    --other   run_0501_decode_special_dp,special_dp_attention

  python3 analyze_kvcache_token_aligned.py \\
    --baseline run_0501_decode_dp_attn,dp_attention \\
    --other   run_0501_decode_special_dp,special_dp_attention \\
    --pairs "0595d0de97924b2386a1e2b92151f550:0847741f65b947ebbe78ed3ded3daf0b:1:0" \\
            "eccd4fafba1741348262ef63ee4d9188:f3a7dacc62644becafba82cc53bb5284:0:0"

``--pairs`` format: left_req:right_req:left_tp:right_tp (omit to use TP-anchor defaults).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PairSpec:
    left_req: str
    right_req: str
    left_tp: int
    right_tp: int


def _load(pt: str) -> dict:
    try:
        return torch.load(pt, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(pt, map_location="cpu")


def top1_token(raw_path: str) -> Optional[int]:
    if not os.path.isfile(raw_path):
        return None
    d = _load(raw_path)
    idx = d.get("topk_idx")
    if idx is None or not torch.is_tensor(idx) or idx.numel() < 1:
        return None
    return int(idx[0].item())


def kv_metrics(path_a: str, path_b: str) -> Tuple[str, Optional[float], Optional[float], bool]:
    if not os.path.isfile(path_a) or not os.path.isfile(path_b):
        return ("missing_kv", None, None, False)
    da, db = _load(path_a), _load(path_b)
    ie = True
    for k in ("kv_indices_head", "kv_indices_tail"):
        ta, tb = da.get(k), db.get(k)
        if torch.is_tensor(ta) and torch.is_tensor(tb):
            ie = ie and bool(torch.equal(ta, tb))
        else:
            ie = ie and ta is None and tb is None
    if da.get("kv_len") != db.get("kv_len"):
        ie = False

    def _mad(x, y):
        if not (torch.is_tensor(x) and torch.is_tensor(y) and x.shape == y.shape):
            return None
        return float((x.float() - y.float()).abs().max().item())

    km = _mad(da.get("key_sample"), db.get("key_sample"))
    vm = _mad(da.get("val_sample"), db.get("val_sample"))
    return ("ok", km, vm, ie)


def _parse_pairs(tokens: Sequence[str]) -> List[PairSpec]:
    out: List[PairSpec] = []
    for t in tokens:
        parts = t.split(":")
        if len(parts) != 4:
            raise ValueError(f"bad --pairs entry {t!r}, want left:right:l_tp:r_tp")
        out.append(PairSpec(parts[0], parts[1], int(parts[2]), int(parts[3])))
    return out


def analyze(
    base: str,
    left_run_dir: str,
    left_tag: str,
    right_run_dir: str,
    right_tag: str,
    *,
    specs: Iterable[PairSpec],
    kv_stage: str = "attn_prepare",
    kv_layer: int = 0,
    margin: float = 0.0,
) -> int:
    kva = os.path.join(base, left_run_dir, "kvcache_fingerprint")
    kvb = os.path.join(base, right_run_dir, "kvcache_fingerprint")
    sa = os.path.join(base, left_run_dir, left_tag)
    sb = os.path.join(base, right_run_dir, right_tag)

    rc = 0
    print(f"BASE={base}")
    print(f"L  run={left_run_dir}  sampling_root={sa}  kv={kva}")
    print(f"R  run={right_run_dir} sampling_root={sb}  kv={kvb}")

    for spec in specs:
        lda = os.path.join(sa, spec.left_req, "decode")
        ldb = os.path.join(sb, spec.right_req, "decode")
        poss = sorted(
            int(fn.split("_", 1)[1])
            for fn in os.listdir(lda)
            if fn.startswith("pos_")
        )
        poss = sorted(
            set(poss)
            & {
                int(fn.split("_", 1)[1])
                for fn in os.listdir(ldb)
                if fn.startswith("pos_")
            }
        )

        print(f"\n== pair L={spec.left_req} R={spec.right_req}  tp=(L{spec.left_tp},R{spec.right_tp}) ==")

        stats_same_k: List[float] = []
        stats_same_v: List[float] = []
        stats_bad = 0

        for pos in poss:
            ra = os.path.join(lda, f"pos_{pos}", f"tp{spec.left_tp}_raw.pt")
            rb = os.path.join(ldb, f"pos_{pos}", f"tp{spec.right_tp}_raw.pt")
            t1_a, t1_b = top1_token(ra), top1_token(rb)
            if t1_a is None or t1_b is None:
                continue
            same_token = t1_a == t1_b
            fk = (
                f"kvcache_layer{kv_layer:04d}_{kv_stage}_tp{spec.left_tp}_pos{pos}_req{spec.left_req}.pt"
            )
            fr = (
                f"kvcache_layer{kv_layer:04d}_{kv_stage}_tp{spec.right_tp}_pos{pos}_req{spec.right_req}.pt"
            )
            st, km, vm, ie = kv_metrics(os.path.join(kva, fk), os.path.join(kvb, fr))
            kv_bad = km is None or vm is None or km > margin or vm > margin
            sig = "=" if same_token else "!="
            kv_tag = (
                "OK"
                if st == "ok" and km is not None and vm is not None and km <= margin and vm <= margin
                else st
            )
            print(f"pos={pos:4d}  top1_same{sig}  kv={kv_tag}  idx_eq={ie}  key_abs={km}  val_abs={vm}")

            if same_token and st == "ok" and km is not None and vm is not None:
                stats_same_k.append(km)
                stats_same_v.append(vm)
                if kv_bad:
                    stats_bad += 1
                    rc = 2

        if stats_same_k:
            print(
                f"summary [same-token steps only]: n={len(stats_same_k)}  "
                f"key_abs max={max(stats_same_k):.6g} mean={sum(stats_same_k)/len(stats_same_k):.6g} | "
                f"val_abs max={max(stats_same_v):.6g} mean={sum(stats_same_v)/len(stats_same_v):.6g}"
            )
        else:
            print("summary [same-token steps only]: none (either no kv or no overlapping decode)")

        if stats_bad:
            print(f"NOTE: same-token KV abs above margin={margin} at {stats_bad} step(s)")

    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        default="/sgl-workspace/sglang/test_scripts/correctness/layer",
        help="Parent directory containing run_* folders.",
    )
    p.add_argument(
        "--baseline",
        required=True,
        help="comma: run_dir_under_base,sampling_tag  e.g. run_0501_decode_tp,tp",
    )
    p.add_argument(
        "--other",
        required=True,
        help="comma: run_dir_under_base,sampling_tag  e.g. run_0501_decode_dp_attn,dp_attention",
    )
    p.add_argument(
        "--pairs",
        nargs="*",
        default=[],
        help="Overrides default TP-anchor pairing: left:right:left_tp:right_tp",
    )
    p.add_argument(
        "--mode",
        choices=["tp_vs_other", "dp_vs_special_presets"],
        default="tp_vs_other",
        help="Preset pairing templates when --pairs omitted.",
    )
    p.add_argument("--kv-stage", default="attn_prepare")
    p.add_argument("--kv-layer", type=int, default=0)
    p.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help='Report nonzero exit if same-token KV max_abs exceeds margin (bf16 granularity often  ~0+) ; default treats >0 strictly.',
    )

    args = p.parse_args(list(argv) if argv is not None else None)
    lr, lt = [x.strip() for x in args.baseline.split(",", 1)]
    rr, rt = [x.strip() for x in args.other.split(",", 1)]

    specs: List[PairSpec]
    if args.pairs:
        specs = _parse_pairs(args.pairs)
    elif args.mode == "dp_vs_special_presets":
        specs = [
            PairSpec(
                left_req="0595d0de97924b2386a1e2b92151f550",
                right_req="0847741f65b947ebbe78ed3ded3daf0b",
                left_tp=1,
                right_tp=0,
            ),
            PairSpec(
                left_req="6121de9639364cabb4405dcdbf3fe258",
                right_req="362a1c7af68d458ea0c1372b92b7f980",
                left_tp=0,
                right_tp=1,
            ),
            PairSpec(
                left_req="eccd4fafba1741348262ef63ee4d9188",
                right_req="f3a7dacc62644becafba82cc53bb5284",
                left_tp=0,
                right_tp=0,
            ),
        ]
    else:
        # Infer right req ids automatically from filenames under sampling root is fragile;
        # keep explicit defaults aligned with historical 0501 runs.
        specs = [
            PairSpec(
                left_req="818abdaace394d039a3e9154d4b00616",
                right_req="0595d0de97924b2386a1e2b92151f550",
                left_tp=1,
                right_tp=1,
            ),  # tp vs dp short
            PairSpec(
                left_req="818abdaace394d039a3e9154d4b00616",
                right_req="0847741f65b947ebbe78ed3ded3daf0b",
                left_tp=0,
                right_tp=0,
            ),  # tp vs sp short (when comparing special)
            PairSpec(
                left_req="cde2b1bc97894270bd68e83f0f62d00f",
                right_req="eccd4fafba1741348262ef63ee4d9188",
                left_tp=0,
                right_tp=0,
            ),  # tp vs dp long — only meaningful when RIGHT is dp
            PairSpec(
                left_req="cde2b1bc97894270bd68e83f0f62d00f",
                right_req="f3a7dacc62644becafba82cc53bb5284",
                left_tp=0,
                right_tp=0,
            ),  # tp vs sp long
        ]

    # Trim specs that mismatch run role in a pragmatic way:
    #
    # - When comparing TP vs DP, RIGHT tag is dp_attention: keep pairs whose right_req is DP.
    # - When comparing TP vs Special, RIGHT is special_dp_attention: filter to right reqs in sp set.
    # Tag must match exactly: substring checks like `"dp_attention" in rt`
    # are wrong because ``special_dp_attention`` contains ``dp_attention``.
    # Trim only for the bundled ``tp_vs_other`` presets: ``dp_vs_special_presets``
    # already names concrete left/right reqs and must keep pairs like DP ``6121..`` ↦ SP ``362a..``.
    if args.mode == "tp_vs_other" and not args.pairs:
        if rt == "special_dp_attention":
            allow_right = {"0847741f65b947ebbe78ed3ded3daf0b", "f3a7dacc62644becafba82cc53bb5284"}
            specs = [s for s in specs if s.right_req in allow_right]
        if rt == "dp_attention":
            allow_right = {"0595d0de97924b2386a1e2b92151f550", "eccd4fafba1741348262ef63ee4d9188"}
            specs = [s for s in specs if s.right_req in allow_right]

    return analyze(
        args.base,
        lr,
        lt,
        rr,
        rt,
        specs=specs,
        kv_stage=args.kv_stage,
        kv_layer=args.kv_layer,
        margin=max(0.0, float(args.margin)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
