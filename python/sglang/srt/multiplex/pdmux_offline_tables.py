"""Load offline PDMUX timing tables (JSON from bench_replay_requests_pdmux)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

_PREFILL_KEY_RE = re.compile(r"^bs=(\d+),max_seq_len=(\d+)$")
_DECODE_KEY_RE = re.compile(r"^bs=(\d+)$")

TieBreak = Literal["up", "down"]


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
    """Lookup prefill/decode times (ms) per stream_group_idx from offline JSON (CPU ms from bench)."""

    def __init__(
        self,
        prefill: Dict[int, Dict[str, float]],
        decode: Dict[int, Dict[str, float]],
        *,
        decode_bs_tie_break: TieBreak = "up",
        prefill_bs_tie_break: TieBreak = "down",
        prefill_max_seq_len_tie_break: TieBreak = "down",
        prefill_launch_only: Optional[Dict[int, Dict[str, float]]] = None,
        decode_gpu_run: Optional[Dict[int, Dict[str, float]]] = None,
    ):
        self.prefill = prefill
        self.decode = decode
        self.decode_bs_tie_break = decode_bs_tie_break
        self.prefill_bs_tie_break = prefill_bs_tie_break
        self.prefill_max_seq_len_tie_break = prefill_max_seq_len_tie_break
        self.prefill_launch_only = prefill_launch_only or {}
        self.decode_gpu_run = decode_gpu_run or {}

    @staticmethod
    def from_json(
        path: str | Path,
        *,
        decode_bs_tie_break: TieBreak = "up",
        prefill_bs_tie_break: TieBreak = "down",
        prefill_max_seq_len_tie_break: TieBreak = "down",
    ) -> PDMuxOfflineTables:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        pre = _parse_stream_group_table(
            raw.get("prefill_cpu_prepare_and_launch_ms")
            or raw.get("prefill_table")
            or {}
        )
        dec = _parse_stream_group_table(
            raw.get("decode_cpu_prepare_and_launch_ms")
            or raw.get("decode_table")
            or {}
        )
        pre_lo = _parse_stream_group_table(
            raw.get("prefill_cpu_launch_only_ms") or {}
        )
        dec_gpu = _parse_stream_group_table(raw.get("decode_gpu_run_ms") or {})
        return PDMuxOfflineTables(
            pre,
            dec,
            decode_bs_tie_break=decode_bs_tie_break,
            prefill_bs_tie_break=prefill_bs_tie_break,
            prefill_max_seq_len_tie_break=prefill_max_seq_len_tie_break,
            prefill_launch_only=pre_lo,
            decode_gpu_run=dec_gpu,
        )

    def lookup_prefill(self, stream_group_idx: int, bs: int, max_seq_len: int) -> Optional[float]:
        """Nearest-neighbor prefill time (ms): exact key, else same-bs nearest max_seq_len, else nearest bs then max_seq_len."""
        return _lookup_prefill_like(
            self.prefill,
            stream_group_idx,
            bs,
            max_seq_len,
            self.prefill_bs_tie_break,
            self.prefill_max_seq_len_tie_break,
        )

    def lookup_prefill_launch_only(
        self, stream_group_idx: int, bs: int, max_seq_len: int
    ) -> Optional[float]:
        """CPU launch-only segment for full-model prefill (ms); same keys as ``lookup_prefill``."""
        return _lookup_prefill_like(
            self.prefill_launch_only,
            stream_group_idx,
            bs,
            max_seq_len,
            self.prefill_bs_tie_break,
            self.prefill_max_seq_len_tie_break,
        )

    def lookup_decode(self, stream_group_idx: int, bs: int) -> Optional[float]:
        """Nearest-neighbor decode **CPU prepare+launch** (ms) by ``bs``."""
        return _lookup_decode_like(
            self.decode, stream_group_idx, bs, self.decode_bs_tie_break
        )

    def lookup_decode_gpu_run(self, stream_group_idx: int, bs: int) -> Optional[float]:
        """Nearest-neighbor decode **GPU run** (ms) by ``bs`` (CUDA event window from bench)."""
        return _lookup_decode_like(
            self.decode_gpu_run, stream_group_idx, bs, self.decode_bs_tie_break
        )


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
