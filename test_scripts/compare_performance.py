#!/usr/bin/env python3
# 需要对比同一个模型，在不同的数据集，不同的配置，不同的请求速率下的表现情况
# 比如在 MODEL_LOG_DIR="deepseek-v2-lite" 中，有多个不同的配置文件夹，"normal_ep_dpattn", "tp_dpattn"
# 每个配置文件夹下，有多个 jsonl 文件，loogle.jsonl, xxxsharegpt.jsonl, xxx1_512.jsonl，用字符串匹配找对应文件
# 对比 p99/p95 ttft、itl、tpot、output_throughput 等，折线图：横轴请求速率，纵轴性能指标
# 不同 jsonl 中字段名不统一，本脚本做兼容解析

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_LOG_DIR = "./out/llama-for-overlap"
# MODEL_LOG_DIR = "./out/deepseek-v3.1-mid/temp"
# MODEL_LOG_DIR = "./out/deepseek-v3.1-mid"
# MODEL_LOG_DIR = "./out/deepseek-v3.1-mini"
# MODEL_LOG_DIR = "./out/temp"
# 性能对比图输出目录。None 表示默认使用 MODEL_LOG_DIR/performance
OUTPUT_DIR: str | None = None
# 数据集匹配规则：文件名包含这些子串即视为对应数据集
DATASET_PATTERNS = {
    "loogle": "loogle",
    "sharegpt": "sharegpt",
    "1_512": "1_512",  # random 1->512
}
# 要绘制的指标：显示名 -> 在各 jsonl 中可能出现的字段名（按优先级尝试）
# 还有 ttft mean, tpot mean, itl mean 这些字段也要考虑
METRIC_KEYS = {
    "p99_ttft_ms": ["p99_ttft_ms", "ttft p99", "p99_ttft", "ttft_p99"],
    "p99_itl_ms": ["p99_itl_ms", "itl p99", "p99_itl", "itl_p99"],
    "p99_tpot_ms": ["p99_tpot_ms", "tpot p99", "p99_tpot", "tpot_p99"],
    "p95_ttft_ms": ["p95_ttft_ms", "ttft p95", "p95_ttft", "ttft_p95"],
    "p95_itl_ms": ["p95_itl_ms", "itl p95", "p95_itl", "itl_p95"],
    "p95_tpot_ms": ["p95_tpot_ms", "tpot p95", "p95_tpot", "tpot_p95"],
    "mean_ttft_ms": ["mean_ttft_ms", "ttft mean", "mean_ttft", "ttft_mean"],
    "mean_itl_ms": ["mean_itl_ms", "itl mean", "mean_itl", "itl_mean"],
    "mean_tpot_ms": ["mean_tpot_ms", "tpot mean", "mean_tpot", "tpot_mean"],
    "output_throughput": ["output_throughput", "output throughput"],
    "total_throughput": ["total_throughput", "total throughput"],
    "request_throughput": ["request_throughput", "request throughput"],
}


def detect_format_and_metrics(line_data: dict) -> tuple[str, dict]:
    """
    检测单行是紧凑格式还是完整格式，并统一提取 request_rate 与各指标值。
    返回 ("compact"|"full", {"request_rate": float, "p99_ttft_ms": float, ...})
    """
    result = {}
    # request_rate
    rr = line_data.get("request_rate")
    if rr is None:
        return None, {}
    if isinstance(rr, str) and rr.lower() == "infinity":
        rr = float("inf")
    else:
        try:
            rr = float(rr)
        except (TypeError, ValueError):
            return None, {}
    result["request_rate"] = rr

    # 各指标：尝试多种字段名，且值可能是字符串需转 float
    for metric_name, key_candidates in METRIC_KEYS.items():
        value = None
        for key in key_candidates:
            if key in line_data:
                raw = line_data[key]
                if raw is None:
                    continue
                if isinstance(raw, str):
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                else:
                    value = float(raw)
                break
        if value is not None:
            result[metric_name] = value

    # 判断格式：有 "ttft mean" 或 "ttft p99"/"ttft p95" 多为紧凑格式
    if "ttft p99" in line_data or "ttft p95" in line_data or "ttft mean" in line_data:
        fmt = "compact"
    else:
        fmt = "full"
    return fmt, result


def load_jsonl(path: Path) -> list[dict]:
    """加载 jsonl，每行解析后得到统一格式的 {'request_rate': r, 'p99_ttft_ms': v, ...} 列表。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            fmt, parsed = detect_format_and_metrics(data)
            if parsed:
                rows.append(parsed)
    return rows


def collect_data(log_dir: str) -> dict:
    """
    扫描 log_dir：每个子目录为一种配置，其下 jsonl 按 DATASET_PATTERNS 归类。
    返回: config_name -> dataset_name -> list of {request_rate, metrics}
    """
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return {}

    out = defaultdict(lambda: defaultdict(list))
    for config_dir in sorted(log_path.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        for jsonl_file in config_dir.glob("*.jsonl"):
            name = jsonl_file.name.lower()
            dataset_name = None
            for label, pattern in DATASET_PATTERNS.items():
                if pattern in name:
                    dataset_name = label
                    break
            if dataset_name is None:
                continue
            rows = load_jsonl(jsonl_file)
            for row in rows:
                out[config_name][dataset_name].append(row)

    # 每个 (config, dataset) 按 request_rate 排序
    for config_name in out:
        for dataset_name in out[config_name]:
            out[config_name][dataset_name].sort(key=lambda x: (x["request_rate"] if x["request_rate"] != float("inf") else 1e9))

    return dict(out)


def _x_inf_placeholder(rows_by_config: dict) -> float:
    """从各配置的 rows 中收集所有有限 request_rate，返回 inf 的占位横坐标。"""
    all_finite = []
    for rows in rows_by_config.values():
        for r in rows:
            rr = r.get("request_rate")
            if rr is not None and rr != float("inf"):
                all_finite.append(rr)
    return (max(all_finite) + 10) if all_finite else 100


def plot_dataset_metric(
    config_rows: dict,
    dataset_name: str,
    metric: str,
    save_path: Path,
    title: str | None = None,
    ylabel: str | None = None,
):
    """
    画一张图：(数据集 × 指标)。横轴 request_rate，纵轴为指标值，每条线代表一种配置。
    config_rows: config_name -> list of {request_rate, metrics}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x_inf = _x_inf_placeholder(config_rows)

    for config_name, rows in config_rows.items():
        if not rows:
            continue
        x, y = [], []
        for r in rows:
            rr = r.get("request_rate")
            val = r.get(metric)
            if rr is None or val is None:
                continue
            if rr == float("inf"):
                rr = x_inf
            x.append(rr)
            y.append(val)
        if x and y:
            ax.plot(x, y, "o-", label=config_name)

    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{dataset_name} - {metric}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def reorganize_by_dataset(data: dict) -> dict:
    """config -> dataset -> rows 转为 dataset -> config -> rows，便于按「数据集×指标」出图。"""
    by_dataset = defaultdict(dict)
    for config_name, datasets in data.items():
        for dataset_name, rows in datasets.items():
            by_dataset[dataset_name][config_name] = rows
    return dict(by_dataset)


def main():
    parser = argparse.ArgumentParser(description="对比同一模型在不同数据集/配置/请求速率下的性能并出图")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        metavar="DIR",
        help="性能对比图输出目录，默认为 MODEL_LOG_DIR/performance",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / MODEL_LOG_DIR
    data = collect_data(str(log_dir))
    if not data:
        print(f"未找到数据，请检查目录: {log_dir}")
        return

    # 按数据集组织：dataset -> config -> rows
    data_by_dataset = reorganize_by_dataset(data)

    # 输出目录：命令行 -o > 模块常量 OUTPUT_DIR > 默认 MODEL_LOG_DIR/performance
    out_dir_raw = args.output_dir if args.output_dir is not None else OUTPUT_DIR
    if out_dir_raw is not None:
        out_dir = Path(out_dir_raw)
        if not out_dir.is_absolute():
            out_dir = script_dir / out_dir
    else:
        out_dir = script_dir / MODEL_LOG_DIR / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    metric_titles = {
        "p99_ttft_ms": "P99 TTFT (ms)",
        "p99_itl_ms": "P99 Inter-token Latency (ms)",
        "p99_tpot_ms": "P99 TPOT (ms)",
        "p95_ttft_ms": "P95 TTFT (ms)",
        "p95_itl_ms": "P95 Inter-token Latency (ms)",
        "p95_tpot_ms": "P95 TPOT (ms)",
        "mean_ttft_ms": "Mean TTFT (ms)",
        "mean_itl_ms": "Mean Inter-token Latency (ms)",
        "mean_tpot_ms": "Mean TPOT (ms)",
        "output_throughput": "Output Throughput (tokens/s)",
        "total_throughput": "Total Throughput (tokens/s)",
        "request_throughput": "Request Throughput (req/s)",
    }
    # 每张图 = 一个数据集 × 一个指标，图中每条线 = 一种配置
    for dataset_name, config_rows in data_by_dataset.items():
        for metric in METRIC_KEYS:
            # 该数据集下是否有该指标的数据（至少一个配置有）
            has_metric = any(
                r.get(metric) is not None
                for rows in config_rows.values()
                for r in rows
            )
            if not has_metric:
                continue
            save_path = out_dir / f"{dataset_name}_{metric}.png"
            ylabel = metric_titles.get(metric, metric)
            title = f"{dataset_name} — {ylabel}"
            plot_dataset_metric(
                config_rows,
                dataset_name,
                metric,
                save_path,
                title=title,
                ylabel=ylabel,
            )
            print(f"已保存: {save_path}")

    # 可选：打印汇总表
    print("\n汇总（每个配置/数据集下各 request_rate 的指标）:")
    for config_name, datasets in data.items():
        for dataset_name, rows in datasets.items():
            if not rows:
                continue
            print(f"  {config_name} / {dataset_name}: {len(rows)} 个速率点")
            for r in rows[:3]:
                rr = r.get("request_rate", "")
                rr_str = "inf" if rr == float("inf") else rr
                print(f"    rate={rr_str} -> {r}")


if __name__ == "__main__":
    main()
