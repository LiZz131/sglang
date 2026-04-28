"""Load offline PDMUX timing tables (JSON from bench_replay_requests_pdmux)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

_PREFILL_KEY_RE = re.compile(r"^bs=(\d+),max_seq_len=(\d+)$")
_DECODE_KEY_RE = re.compile(r"^bs=(\d+)$")

TieBreak = Literal["up", "down"]
MetricKind = Literal["gpu_run_ms", "cpu_prepare_and_launch_ms", "cpu_launch_only_ms"]


def _tie_pick(target: int, candidates: Sequence[int], tie: TieBreak) -> Optional[int]:
    """Pick the closest value to ``target``; on equal distance, prefer larger (up) or smaller (down)."""
    uniq = sorted(set(candidates))
    if not uniq:
        return None
    best_d = min(abs(target - x) for x in uniq)
    tied = [x for x in uniq if abs(target - x) == best_d]
    if tie == "up":
        return max(tied)
    return min(tied)


def _tie_pick_value(
    target: int,
    pairs: Sequence[Tuple[int, float]],
    tie: TieBreak,
) -> Optional[float]:
    """``pairs`` are (coord, value); pick coord closest to ``target``, tie-break, return its value."""
    if not pairs:
        return None
    coords = [p[0] for p in pairs]
    chosen = _tie_pick(target, coords, tie)
    if chosen is None:
        return None
    same = [p for p in pairs if p[0] == chosen]
    return float(same[0][1])


def _lookup_prefill_like(
    table_by_sg: Dict[int, Dict[str, float]],
    stream_group_idx: int,
    bs: int,
    max_seq_len: int,
    prefill_bs_tie_break: TieBreak,
    prefill_max_seq_len_tie_break: TieBreak,
) -> Optional[float]:
    table = table_by_sg.get(stream_group_idx)
    if not table:
        return None
    exact = f"bs={bs},max_seq_len={max_seq_len}"
    if exact in table:
        return float(table[exact])
    rows: List[Tuple[int, int, float]] = []
    for key, val in table.items():
        m = _PREFILL_KEY_RE.match(key)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), float(val)))
    if not rows:
        return None
    same_bs = [(msl, v) for (b, msl, v) in rows if b == bs]
    if same_bs:
        return _tie_pick_value(max_seq_len, same_bs, prefill_max_seq_len_tie_break)
    distinct_bs = sorted({b for (b, _msl, _v) in rows})
    picked_bs = _tie_pick(bs, distinct_bs, prefill_bs_tie_break)
    if picked_bs is None:
        return None
    msl_pairs = [(msl, v) for (b, msl, v) in rows if b == picked_bs]
    return _tie_pick_value(max_seq_len, msl_pairs, prefill_max_seq_len_tie_break)


def _lookup_decode_like(
    table_by_sg: Dict[int, Dict[str, float]],
    stream_group_idx: int,
    bs: int,
    decode_bs_tie_break: TieBreak,
) -> Optional[float]:
    table = table_by_sg.get(stream_group_idx)
    if not table:
        return None
    key = f"bs={bs}"
    if key in table:
        return float(table[key])
    bs_list: List[int] = []
    for k in table:
        m = _DECODE_KEY_RE.match(k)
        if m:
            bs_list.append(int(m.group(1)))
    if not bs_list:
        return None
    picked = _tie_pick(bs, bs_list, decode_bs_tie_break)
    if picked is None:
        return None
    return float(table[f"bs={picked}"])


class PDMuxOfflineTables:
    """Lookup prefill/decode times (ms) per stream_group_idx from offline JSON.

    Each of prefill/decode supports three metric kinds:
    - ``gpu_run_ms``: CUDA-event elapsed time of ``forward`` (bench key: ``*_gpu_run_ms``)
    - ``cpu_prepare_and_launch_ms``: CPU interval including prepare+launch (bench key: ``*_cpu_prepare_and_launch_ms``)
    - ``cpu_launch_only_ms``: CPU launch-only interval (bench key: ``*_cpu_launch_only_ms``)
    """

    def __init__(
        self,
        prefill_gpu_run: Dict[int, Dict[str, float]],
        prefill_cpu_prepare_and_launch: Dict[int, Dict[str, float]],
        prefill_cpu_launch_only: Dict[int, Dict[str, float]],
        decode_gpu_run: Dict[int, Dict[str, float]],
        decode_cpu_prepare_and_launch: Dict[int, Dict[str, float]],
        decode_cpu_launch_only: Dict[int, Dict[str, float]],
        *,
        decode_bs_tie_break: TieBreak = "up",
        prefill_bs_tie_break: TieBreak = "down",
        prefill_max_seq_len_tie_break: TieBreak = "down",
    ):
        self.prefill_gpu_run = prefill_gpu_run
        self.prefill_cpu_prepare_and_launch = prefill_cpu_prepare_and_launch
        self.prefill_cpu_launch_only = prefill_cpu_launch_only
        self.decode_gpu_run = decode_gpu_run
        self.decode_cpu_prepare_and_launch = decode_cpu_prepare_and_launch
        self.decode_cpu_launch_only = decode_cpu_launch_only
        self.decode_bs_tie_break = decode_bs_tie_break
        self.prefill_bs_tie_break = prefill_bs_tie_break
        self.prefill_max_seq_len_tie_break = prefill_max_seq_len_tie_break

    @staticmethod
    def from_json(
        path: str | Path,
        *,
        decode_bs_tie_break: TieBreak = "up",
        prefill_bs_tie_break: TieBreak = "down",
        prefill_max_seq_len_tie_break: TieBreak = "down",
    ) -> PDMuxOfflineTables:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        # Primary schema (bench_replay_requests_pdmux)
        pre_gpu = _parse_stream_group_table(raw.get("prefill_gpu_run_ms") or {})
        pre_pl = _parse_stream_group_table(
            raw.get("prefill_cpu_prepare_and_launch_ms")
            or raw.get("prefill_table")
            or {}
        )
        pre_lo = _parse_stream_group_table(raw.get("prefill_cpu_launch_only_ms") or {})
        dec_gpu = _parse_stream_group_table(raw.get("decode_gpu_run_ms") or {})
        dec_pl = _parse_stream_group_table(
            raw.get("decode_cpu_prepare_and_launch_ms")
            or raw.get("decode_table")
            or {}
        )
        dec_lo = _parse_stream_group_table(raw.get("decode_cpu_launch_only_ms") or {})

        return PDMuxOfflineTables(
            pre_gpu,
            pre_pl,
            pre_lo,
            dec_gpu,
            dec_pl,
            dec_lo,
            decode_bs_tie_break=decode_bs_tie_break,
            prefill_bs_tie_break=prefill_bs_tie_break,
            prefill_max_seq_len_tie_break=prefill_max_seq_len_tie_break,
        )

    def lookup_prefill(
        self,
        stream_group_idx: int,
        bs: int,
        max_seq_len: int,
        *,
        metric: MetricKind = "gpu_run_ms",
    ) -> Optional[float]:
        """Nearest-neighbor prefill time (ms) for a given metric kind."""
        if metric == "gpu_run_ms":
            table = self.prefill_gpu_run
        elif metric == "cpu_prepare_and_launch_ms":
            table = self.prefill_cpu_prepare_and_launch
        elif metric == "cpu_launch_only_ms":
            table = self.prefill_cpu_launch_only
        else:
            return None
        return _lookup_prefill_like(
            table,
            stream_group_idx,
            bs,
            max_seq_len,
            self.prefill_bs_tie_break,
            self.prefill_max_seq_len_tie_break,
        )

    def lookup_decode(
        self,
        stream_group_idx: int,
        bs: int,
        *,
        metric: MetricKind = "gpu_run_ms",
    ) -> Optional[float]:
        """Nearest-neighbor decode time (ms) for a given metric kind."""
        if metric == "gpu_run_ms":
            table = self.decode_gpu_run
        elif metric == "cpu_prepare_and_launch_ms":
            table = self.decode_cpu_prepare_and_launch
        elif metric == "cpu_launch_only_ms":
            table = self.decode_cpu_launch_only
        else:
            return None
        return _lookup_decode_like(table, stream_group_idx, bs, self.decode_bs_tie_break)


def _parse_stream_group_table(raw: dict) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for sg_key, inner in raw.items():
        try:
            sg = int(sg_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(inner, dict):
            continue
        out[sg] = {str(k): float(v) for k, v in inner.items()}
    return out


def load_pdmux_offline_tables(
    path: str | Path,
    *,
    decode_bs_tie_break: TieBreak = "up",
    prefill_bs_tie_break: TieBreak = "down",
    prefill_max_seq_len_tie_break: TieBreak = "down",
) -> PDMuxOfflineTables:
    return PDMuxOfflineTables.from_json(
        path,
        decode_bs_tie_break=decode_bs_tie_break,
        prefill_bs_tie_break=prefill_bs_tie_break,
        prefill_max_seq_len_tie_break=prefill_max_seq_len_tie_break,
    )


def normalize_tie_break(s: str, default: TieBreak) -> TieBreak:
    x = (s or "").strip().lower()
    if x in ("up", "down"):
        return x  # type: ignore[return-value]
    return default
