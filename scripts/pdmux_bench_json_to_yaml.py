#!/usr/bin/env python3
"""
将 bench 建模产出的 JSON 转为 YAML，便于写入配置或接入 sglang（如 pdmux_time_model）。

用法::

  # 1) 任意 JSON -> YAML（便于人工阅读）
  python3 pdmux_bench_json_to_yaml.py bench_model_results.json -o bench_model_results.yaml

    ../test_scripts/modeling/out/15-layer-originweight/new/out

  # 2) 合并 modeling 输出目录（默认极简：仅 prefill_sm / decode_sm + 四个系数三元组）
  python3 pdmux_bench_json_to_yaml.py --merge-dir /path/to/out/15-layer -o pdmux_fitted.yaml

  # 3) 需要原先的完整合并（batch_index、runs_by_stem、指标等）
  python3 pdmux_bench_json_to_yaml.py --merge-dir ... -o pdmux_full.yaml --full

系数顺序（与 pdmux_time_model 一致）::
  - prefill_launch / prefill_run: (sum_n^2, sum_n, bias)
  - decode_launch / decode_run: (L, sum(global_num_tokens), bias)

依赖: PyYAML（pip install pyyaml）
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise SystemExit("需要 PyYAML: pip install pyyaml") from e


class _NoAliasDumper(yaml.SafeDumper):
    """避免 YAML 对重复子树使用 &锚点 / *引用，便于人工编辑与 diff。"""

    def ignore_aliases(self, data):  # type: ignore[override]
        return True


# p104_d28_pdmux_bench_result_prefill -> (p104_d28, prefill|decode)
RE_STEM = re.compile(
    r"^(?P<pair>p\d+_d\d+)_pdmux_bench_result_(?P<stage>prefill|decode)$",
    re.I,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_to_yaml(data: Any, path: Path, *, default_flow_style: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        data,
        Dumper=_NoAliasDumper,
        allow_unicode=True,
        default_flow_style=default_flow_style,
        sort_keys=False,
    )
    path.write_text(text, encoding="utf-8")


def _coeff_triple(block: dict[str, Any] | None) -> list[float] | None:
    if not block or "coeffs" not in block:
        return None
    c = block["coeffs"]
    if isinstance(c, list) and len(c) == 3:
        return [float(x) for x in c]
    return None


def _strip_metrics_for_coeffs_only(node: Any) -> Any:
    """递归删除拟合报告中的样本量、R2 等，仅保留 coeffs（若存在）。"""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "coeffs":
                out[k] = v
            elif k == "stream_groups":
                out[k] = _strip_metrics_for_coeffs_only(v)
            elif isinstance(v, dict) and "coeffs" in v:
                out[k] = {"coeffs": v["coeffs"]}
            else:
                out[k] = _strip_metrics_for_coeffs_only(v)
        return out
    if isinstance(node, list):
        return [_strip_metrics_for_coeffs_only(x) for x in node]
    return node


def merge_modeling_dir(
    merge_dir: Path,
    *,
    coeffs_only: bool,
) -> dict[str, Any]:
    """
    读取 merge_dir/batch_index.json（若存在），并合并各子目录 bench_model_results.json。
    同时按 pXX_dYY 将 prefill / decode 两类文件合并到同一键下。
    """
    merge_dir = merge_dir.resolve()
    index_path = merge_dir / "batch_index.json"
    merged: dict[str, Any] = {
        "meta": {
            "merge_dir": str(merge_dir),
            "batch_index": str(index_path) if index_path.is_file() else None,
        },
        "runs_by_stem": {},
        "by_sm_pair": {},
    }

    if index_path.is_file():
        merged["meta"]["batch_index_content"] = load_json(index_path)

    # 扫描子目录中的 bench_model_results.json
    for sub in sorted(merge_dir.iterdir()):
        if not sub.is_dir():
            continue
        br = sub / "bench_model_results.json"
        if not br.is_file():
            continue
        stem = sub.name
        data = load_json(br)
        if coeffs_only:
            data = _strip_metrics_for_coeffs_only(data)
        merged["runs_by_stem"][stem] = data

        m = RE_STEM.match(stem)
        if not m:
            continue
        pair = m.group("pair")
        stage = m.group("stage")
        bucket = merged["by_sm_pair"].setdefault(
            pair,
            {"prefill_sm": None, "decode_sm": None, "prefill": None, "decode": None},
        )
        # 从 bench_meta 取 SM（若存在）
        bm = data.get("bench_meta") or {}
        if bm.get("prefill_sm") is not None:
            bucket["prefill_sm"] = bm["prefill_sm"]
        if bm.get("decode_sm") is not None:
            bucket["decode_sm"] = bm["decode_sm"]
        if pair.startswith("p") and "_d" in pair:
            try:
                ps, ds = pair[1:].split("_d", 1)
                bucket["prefill_sm"] = bucket["prefill_sm"] or int(ps)
                bucket["decode_sm"] = bucket["decode_sm"] or int(ds)
            except (ValueError, TypeError):
                pass
        bucket[stage] = {
            "stream_groups": data.get("stream_groups", {}),
            "exclude_outliers": data.get("exclude_outliers"),
            "did_prefill": data.get("did_prefill"),
            "did_decode": data.get("did_decode"),
        }

    # 可选：为每个 sm 对生成与 PerGroupTimeCoeffs 对齐的扁平系数（按 stream_group 字符串键）
    per_group_like: dict[str, Any] = {}
    for pair, b in merged["by_sm_pair"].items():
        sg_map: dict[str, Any] = {}
        pre = b.get("prefill") or {}
        dec = b.get("decode") or {}
        pre_sg = (pre.get("stream_groups") or {}) if isinstance(pre, dict) else {}
        dec_sg = (dec.get("stream_groups") or {}) if isinstance(dec, dict) else {}

        all_sg = set(pre_sg.keys()) | set(dec_sg.keys())
        for sg in sorted(all_sg, key=lambda x: int(x)):
            entry: dict[str, Any] = {}
            for name, src in (
                ("prefill_launch", pre_sg.get(sg, {}).get("prefill_launch_ms")),
                ("prefill_run", pre_sg.get(sg, {}).get("prefill_run_ms")),
                ("decode_launch", dec_sg.get(sg, {}).get("decode_launch_ms")),
                ("decode_run", dec_sg.get(sg, {}).get("decode_run_ms")),
            ):
                t = _coeff_triple(src if isinstance(src, dict) else None)
                if t is not None:
                    entry[name] = t
            if entry:
                sg_map[sg] = entry
        if sg_map:
            per_group_like[pair] = {
                "prefill_sm": b.get("prefill_sm"),
                "decode_sm": b.get("decode_sm"),
                "stream_groups": sg_map,
            }
    merged["per_group_triples_for_pdmux_time_model"] = per_group_like

    return merged


def _four_triples_for_sg(
    pre_sg: dict[str, Any],
    dec_sg: dict[str, Any],
    sg: str,
) -> dict[str, list[float]] | None:
    entry: dict[str, list[float]] = {}
    for name, src in (
        ("prefill_launch", pre_sg.get(sg, {}).get("prefill_launch_ms")),
        ("prefill_run", pre_sg.get(sg, {}).get("prefill_run_ms")),
        ("decode_launch", dec_sg.get(sg, {}).get("decode_launch_ms")),
        ("decode_run", dec_sg.get(sg, {}).get("decode_run_ms")),
    ):
        t = _coeff_triple(src if isinstance(src, dict) else None)
        if t is not None:
            entry[name] = t
    return entry if len(entry) == 4 else None


def merge_modeling_dir_minimal(merge_dir: Path) -> dict[str, Any]:
    """
    仅输出每个 pXX_dYY 下：
      prefill_sm, decode_sm, prefill_launch, prefill_run, decode_launch, decode_run
    多个 stream_group 时多一层 ``stream_groups: { "0": {...}, "1": {...} }``。
    """
    merge_dir = merge_dir.resolve()
    by_pair: dict[str, dict[str, Any]] = {}

    for sub in sorted(merge_dir.iterdir()):
        if not sub.is_dir():
            continue
        br = sub / "bench_model_results.json"
        if not br.is_file():
            continue
        stem = sub.name
        data = load_json(br)
        m = RE_STEM.match(stem)
        if not m:
            continue
        pair = m.group("pair")
        stage = m.group("stage")
        bucket = by_pair.setdefault(
            pair,
            {"prefill_sm": None, "decode_sm": None, "prefill": None, "decode": None},
        )
        bm = data.get("bench_meta") or {}
        if bm.get("prefill_sm") is not None:
            bucket["prefill_sm"] = bm["prefill_sm"]
        if bm.get("decode_sm") is not None:
            bucket["decode_sm"] = bm["decode_sm"]
        if pair.startswith("p") and "_d" in pair:
            try:
                ps, ds = pair[1:].split("_d", 1)
                bucket["prefill_sm"] = bucket["prefill_sm"] if bucket["prefill_sm"] is not None else int(ps)
                bucket["decode_sm"] = bucket["decode_sm"] if bucket["decode_sm"] is not None else int(ds)
            except (ValueError, TypeError):
                pass
        bucket[stage] = {
            "stream_groups": data.get("stream_groups", {}),
        }

    out: dict[str, Any] = {}
    for pair, b in sorted(by_pair.items()):
        pre = b.get("prefill") or {}
        dec = b.get("decode") or {}
        pre_sg = (pre.get("stream_groups") or {}) if isinstance(pre, dict) else {}
        dec_sg = (dec.get("stream_groups") or {}) if isinstance(dec, dict) else {}
        all_sg = sorted(set(pre_sg.keys()) | set(dec_sg.keys()), key=lambda x: int(x))

        psm = b.get("prefill_sm")
        dsm = b.get("decode_sm")
        if psm is None or dsm is None:
            continue
        if not all_sg:
            continue

        if len(all_sg) == 1:
            sg = all_sg[0]
            four = _four_triples_for_sg(pre_sg, dec_sg, sg)
            if not four:
                continue
            out[pair] = {
                "prefill_sm": int(psm),
                "decode_sm": int(dsm),
                **four,
            }
        else:
            groups: dict[str, Any] = {}
            for sg in all_sg:
                four = _four_triples_for_sg(pre_sg, dec_sg, sg)
                if four and len(four) == 4:
                    groups[str(sg)] = four
            if not groups:
                continue
            out[pair] = {
                "prefill_sm": int(psm),
                "decode_sm": int(dsm),
                "stream_groups": groups,
            }

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=None,
        help="单个 JSON 文件（与 --merge-dir 二选一）",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="输出 YAML 路径",
    )
    ap.add_argument(
        "--merge-dir",
        type=Path,
        default=None,
        help="model_pdmux_bench_json 批量输出目录（含 batch_index.json 与子目录）",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="合并时输出完整结构（meta、runs_by_stem、指标等）；默认仅输出极简系数表",
    )
    ap.add_argument(
        "--coeffs-only",
        action="store_true",
        help="与 --full 联用：在完整合并中去掉 R² 等非 coeffs 字段（默认 --full 未开启时无效果）",
    )
    ap.add_argument(
        "--flow-style",
        action="store_true",
        help="YAML 中尽量使用流式（紧凑）风格",
    )
    args = ap.parse_args()

    if args.merge_dir is not None:
        merge_path = args.merge_dir.expanduser().resolve()
        if args.full:
            data = merge_modeling_dir(merge_path, coeffs_only=args.coeffs_only)
        else:
            data = merge_modeling_dir_minimal(merge_path)
        json_to_yaml(data, args.output.expanduser().resolve(), default_flow_style=args.flow_style)
        print(f"Wrote {args.output}")
        return

    if args.json_path is None:
        raise SystemExit("请提供 json_path，或使用 --merge-dir")

    jp = args.json_path.expanduser().resolve()
    if not jp.is_file():
        raise SystemExit(f"文件不存在: {jp}")
    data = load_json(jp)
    json_to_yaml(data, args.output.expanduser().resolve(), default_flow_style=args.flow_style)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
