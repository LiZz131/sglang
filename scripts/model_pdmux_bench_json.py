#!/usr/bin/env python3
"""
对 bench_one_batch_pdmux 写出的 JSON（见 python/sglang/bench_one_batch_pdmux.py）做与 model_pdmux_timing.py 一致的线性建模与可视化。

JSON 结构（rank0 写出）::
    {
      "run_name": ...,
      "stream_group_idx": ...,
      "configs": [
        {
          "summary": { "stream_group_idx", "prefill_sm", "decode_sm", "batch_size", ... },
          "prefill_records": [ { "batch_info", "prefill_launch_ms", "prefill_run_ms", ... }, ... ],
          "decode_records": [ { "batch_info", "decode_launch_ms", "decode_run_ms", ... }, ... ],
        },
        ...
      ]
    }

典型用法（与 test_scripts/modeling 下多组 p*_d*_pdmux_bench_result_{prefill,decode}.json 一致）::
  - prefill / decode 分文件保存时，一个文件里往往只有一类 records 非空；
    脚本会按数据自动跳过空阶段，无需再写 --skip-decode / --skip-prefill。

  # 单个文件
  python3 model_pdmux_bench_json.py p80_d52_pdmux_bench_result_prefill.json

  # 批量：多个路径 + glob（可混用）
  python3 model_pdmux_bench_json.py -g 'p*_pdmux_bench_result_prefill.json' -g 'p*_pdmux_bench_result_decode.json' -o ./out

    ../test_scripts/modeling/out/15-layer-originweight/new/

  # 输出：单文件 -> -o 或 同目录/pdmux_bench_model；
         多文件 -> -o/<各 json 的 stem>/（未指定 -o 时默认为 首个文件所在目录/pdmux_bench_model_runs/<stem>/）

模型（与 notes/todos 一致）::
  - Prefill:  T = a1 * sum(n^2) + a2 * sum(n) + a3   （目标为 launch 或 run 的 ms）
  - Decode:   T = b1 * sum(local) + b2 * bs + b3     （目标为 launch 或 run 的 ms）

异常值：复用 model_pdmux_timing 中 MAD 残差迭代剔除；可用 --no-exclude-outliers 关闭。
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 与同目录下 model_pdmux_timing 共享解析与拟合
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import model_pdmux_timing as mpt


def parse_batch_info_decode_compat(
    batch_info: str,
) -> tuple[list[int], list[int], list[int] | None] | None:
    """
    兼容新的 decode batch_info 格式：

    - 旧：\"[local...]|[global...]\"\n
    - 新：\"[local...]|[global...]|[global_seq_lens_sum_per_dp...]\"\n

    返回 (local, global, per_dp_sum_or_None)。空字符串返回 None。
    """
    s = (batch_info or "").strip()
    if s == "":
        return None
    parts = s.split("|")
    if len(parts) == 1:
        return mpt.parse_int_list(parts[0]), [], None
    if len(parts) == 2:
        return mpt.parse_int_list(parts[0]), mpt.parse_int_list(parts[1]), None
    # len(parts) >= 3：只取前三段，后续若再扩展不影响建模
    return (
        mpt.parse_int_list(parts[0]),
        mpt.parse_int_list(parts[1]),
        mpt.parse_int_list(parts[2]),
    )


def flatten_bench_payload(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """将每个 config 的 summary 字段并入每条 record。"""
    prefill_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    for cfg in data.get("configs", []):
        summ = cfg.get("summary") or {}
        sg = summ.get("stream_group_idx")
        extra = {
            "stream_group_idx": sg,
            "prefill_sm": summ.get("prefill_sm"),
            "decode_sm": summ.get("decode_sm"),
            "config_batch_size": summ.get("batch_size"),
            "config_input_len": summ.get("input_len"),
            "config_output_len": summ.get("output_len"),
        }
        for r in cfg.get("prefill_records") or []:
            prefill_rows.append({**r, **extra})
        for r in cfg.get("decode_records") or []:
            decode_rows.append({**r, **extra})
    return prefill_rows, decode_rows


def build_prefill_df_ms(
    records: list[dict[str, Any]], *, target_ms_key: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in records:
        bi = mpt.parse_batch_info_prefill(r.get("batch_info", ""))
        if bi is None:
            continue
        if r.get(target_ms_key) is None:
            continue
        sum_n = int(sum(bi))
        sum_n2 = int(sum(x * x for x in bi))
        rows.append(
            {
                "y": float(r[target_ms_key]),
                "stream_group_idx": r.get("stream_group_idx"),
                "sum_n": sum_n,
                "sum_n2": sum_n2,
                "bias": 1.0,
                "batch_info": r.get("batch_info", ""),
                "config_batch_size": r.get("config_batch_size"),
                "config_input_len": r.get("config_input_len"),
                "config_output_len": r.get("config_output_len"),
            }
        )
    return pd.DataFrame(rows)


def build_decode_df_ms(
    records: list[dict[str, Any]], *, target_ms_key: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in records:
        bi = parse_batch_info_decode_compat(r.get("batch_info", ""))
        if bi is None:
            continue
        if r.get(target_ms_key) is None:
            continue
        local, global_, gsspd = bi
        sum_local = int(sum(local))
        bs = int(sum(global_))
        gsspd_sum = int(sum(gsspd)) if gsspd else None
        gsspd_max = int(max(gsspd)) if gsspd else None
        rows.append(
            {
                "y": float(r[target_ms_key]),
                "stream_group_idx": r.get("stream_group_idx"),
                "sum_local": sum_local,
                "bs": bs,
                # 兼容字段：先写入 CSV/DF，建模可暂时不用
                "global_seq_lens_sum_per_dp": gsspd,
                "global_seq_lens_sum_per_dp_sum": gsspd_sum,
                "global_seq_lens_sum_per_dp_max": gsspd_max,
                "bias": 1.0,
                "batch_info": r.get("batch_info", ""),
                "decode_step": r.get("decode_step"),
                "config_batch_size": r.get("config_batch_size"),
                "config_input_len": r.get("config_input_len"),
                "config_output_len": r.get("config_output_len"),
            }
        )
    return pd.DataFrame(rows)


def fit_by_group_bench(
    df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    out_dir: Path,
    *,
    max_iter: int,
    mad_k: float,
    exclude_outliers: bool,
    results: dict[str, Any],
) -> None:
    if df.empty:
        results["global"][name] = {"empty": True, "reason": "no_rows_after_parse"}
        return
    df = df.copy()
    df["stream_group_idx"] = df["stream_group_idx"].fillna(-1).astype(int)
    for sg, gdf in df.groupby("stream_group_idx"):
        if sg == -1:
            continue
        gdf = gdf.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"] + feature_cols)
        if len(gdf) < len(feature_cols):
            continue

        rep, out = mpt.fit_one(
            gdf,
            feature_cols,
            "y",
            max_iter=max_iter,
            mad_k=mad_k,
            exclude_outliers=exclude_outliers,
        )
        results["stream_groups"].setdefault(str(int(sg)), {})[name] = asdict(rep)

        suffix = "" if exclude_outliers else "_all"
        title = f"{name} (stream_group={sg}, bench ms){suffix}"
        mpt.plot_fit(
            out,
            title=title,
            out_png=out_dir / f"{name}_sg{sg}{suffix}.png",
            value_unit="ms",
        )
        out.to_csv(out_dir / f"{name}_sg{sg}{suffix}.csv", index=False)


def collect_input_paths(
    positional: list[Path],
    glob_patterns: list[str],
) -> list[Path]:
    paths: list[Path] = []
    for p in positional:
        p = p.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"文件不存在: {p}")
        paths.append(p)
    for pattern in glob_patterns:
        matches = sorted(glob_mod.glob(pattern, recursive=True))
        if not matches:
            raise SystemExit(f"--glob 未匹配任何文件: {pattern!r}")
        for m in matches:
            pp = Path(m).expanduser().resolve()
            if pp.is_file() and pp.suffix.lower() == ".json":
                paths.append(pp)
    # 去重且保序
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def resolve_out_dir_for_file(
    in_path: Path,
    *,
    batch_size: int,
    output: Path | None,
) -> Path:
    """单文件：output 或 同目录/pdmux_bench_model；多文件：output 或 同目录/pdmux_bench_model_runs/<stem>。"""
    if batch_size == 1:
        return (output or (in_path.parent / "pdmux_bench_model")).expanduser().resolve()
    root = (output or (in_path.parent / "pdmux_bench_model_runs")).expanduser().resolve()
    return root / in_path.stem


def run_one_json(
    in_path: Path,
    out_dir: Path,
    *,
    max_iter: int,
    mad_k: float,
    exclude_outliers: bool,
    skip_prefill: bool,
    skip_decode: bool,
) -> dict[str, Any]:
    mpt.ensure_dir(out_dir)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pre_r, dec_r = flatten_bench_payload(data)
    # 分文件 bench 时通常只有一类有数据：自动跳过空侧（可被显式 --skip-* 覆盖）
    auto_skip_prefill = len(pre_r) == 0
    auto_skip_decode = len(dec_r) == 0
    do_prefill = not skip_prefill and not auto_skip_prefill
    do_decode = not skip_decode and not auto_skip_decode

    results: dict[str, Any] = {
        "source_json": str(in_path),
        "bench_meta": {
            "run_name": data.get("run_name"),
            "stream_group_idx_top": data.get("stream_group_idx"),
            "model_path": data.get("model_path"),
            "tp_size": data.get("tp_size"),
            "dp_size": data.get("dp_size"),
            "sm_counts": data.get("sm_counts"),
        },
        "exclude_outliers": exclude_outliers,
        "prefill_record_rows": len(pre_r),
        "decode_record_rows": len(dec_r),
        "auto_skip_prefill": auto_skip_prefill,
        "auto_skip_decode": auto_skip_decode,
        "did_prefill": do_prefill,
        "did_decode": do_decode,
        "stream_groups": {},
        "global": {},
    }

    if do_prefill:
        df_pl = build_prefill_df_ms(pre_r, target_ms_key="prefill_launch_ms")
        df_pr = build_prefill_df_ms(pre_r, target_ms_key="prefill_run_ms")
        fit_by_group_bench(
            df_pl,
            ["sum_n2", "sum_n", "bias"],
            "prefill_launch_ms",
            out_dir,
            max_iter=max_iter,
            mad_k=mad_k,
            exclude_outliers=exclude_outliers,
            results=results,
        )
        fit_by_group_bench(
            df_pr,
            ["sum_n2", "sum_n", "bias"],
            "prefill_run_ms",
            out_dir,
            max_iter=max_iter,
            mad_k=mad_k,
            exclude_outliers=exclude_outliers,
            results=results,
        )
    else:
        results["global"]["prefill_launch_ms"] = {"skipped": True}
        results["global"]["prefill_run_ms"] = {"skipped": True}

    if do_decode:
        df_dl = build_decode_df_ms(dec_r, target_ms_key="decode_launch_ms")
        df_dr = build_decode_df_ms(dec_r, target_ms_key="decode_run_ms")
        fit_by_group_bench(
            df_dl,
            ["sum_local", "bs", "bias"],
            "decode_launch_ms",
            out_dir,
            max_iter=max_iter,
            mad_k=mad_k,
            exclude_outliers=exclude_outliers,
            results=results,
        )
        fit_by_group_bench(
            df_dr,
            ["sum_local", "bs", "bias"],
            "decode_run_ms",
            out_dir,
            max_iter=max_iter,
            mad_k=mad_k,
            exclude_outliers=exclude_outliers,
            results=results,
        )
    else:
        results["global"]["decode_launch_ms"] = {"skipped": True}
        results["global"]["decode_run_ms"] = {"skipped": True}

    out_json = out_dir / "bench_model_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return {"out_json": str(out_json), "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "bench_json",
        nargs="*",
        type=Path,
        default=[],
        help="一个或多个 bench JSON 路径",
    )
    ap.add_argument(
        "--glob",
        "-g",
        action="append",
        default=[],
        metavar="PATTERN",
        help="glob 模式（相对当前工作目录或可带路径），可重复，例如 -g 'p*_pdmux_bench_result_prefill.json'",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="单文件：输出目录；多文件：作为父目录，每个 JSON 写入 <output>/<stem>/",
    )
    ap.add_argument("--mad-k", type=float, default=3.5)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument(
        "--no-exclude-outliers",
        action="store_true",
        help="关闭 MAD 残差剔除，对全部样本做一次 OLS",
    )
    ap.add_argument(
        "--skip-decode",
        action="store_true",
        help="强制不拟合 decode（即使 decode_records 非空）",
    )
    ap.add_argument(
        "--skip-prefill",
        action="store_true",
        help="强制不拟合 prefill（即使 prefill_records 非空）",
    )
    ap.add_argument(
        "--index-json",
        type=Path,
        default=None,
        help="批量时可选：将各文件的 bench_model_results.json 路径写入该汇总 JSON",
    )
    args = ap.parse_args()

    in_paths = collect_input_paths(list(args.bench_json), list(args.glob))
    if not in_paths:
        raise SystemExit("请提供至少一个 JSON：位置参数 或 --glob")

    exclude_outliers = not args.no_exclude_outliers
    batch_n = len(in_paths)
    index_entries: list[dict[str, Any]] = []

    for in_path in in_paths:
        out_dir = resolve_out_dir_for_file(
            in_path, batch_size=batch_n, output=args.output
        )
        summary = run_one_json(
            in_path,
            out_dir,
            max_iter=args.max_iter,
            mad_k=args.mad_k,
            exclude_outliers=exclude_outliers,
            skip_prefill=args.skip_prefill,
            skip_decode=args.skip_decode,
        )
        print(f"Wrote {summary['out_json']}")
        index_entries.append(
            {
                "source_json": str(in_path),
                "output_dir": str(out_dir),
                "bench_model_results": summary["out_json"],
            }
        )

    if batch_n > 1 or args.index_json:
        idx_path = (
            args.index_json
            if args.index_json is not None
            else (
                (args.output or in_paths[0].parent / "pdmux_bench_model_runs").expanduser().resolve()
                / "batch_index.json"
            )
        )
        if batch_n == 1 and args.index_json:
            idx_path = args.index_json.expanduser().resolve()
        mpt.ensure_dir(idx_path.parent)
        payload = {
            "exclude_outliers": exclude_outliers,
            "count": len(index_entries),
            "runs": index_entries,
        }
        with idx_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote {idx_path}")


if __name__ == "__main__":
    main()
