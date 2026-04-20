"""
PD-Multiplexing time prediction for special-DP-attention stream group selection.

Formulas (per stream_group index g, coefficients only differ between g and between phases):
  prefill:  T = c1 * sum(n_i^2) + c2 * sum(n_i) + c3   (n_i = extend_input_len)
  decode:   T = c1 * L + c2 * B + c3
            L = max(global_seq_lens_sum_per_dp), B = sum(global_num_tokens)

Fitted YAML: each entry (any top-level key) must contain prefill_sm, decode_sm (ints) and
prefill_launch/run, decode_launch/run as length-3 lists. Matching to runtime stream groups
uses (prefill_sm, decode_sm) from sm_counts, not key names. If no exact match, nearest
neighbor in (prefill_sm, decode_sm) space is used.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import yaml

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

# (c1, c2, c3) for linear features (x, y) -> c1*x + c2*y + c3
Triple = Tuple[float, float, float]


@dataclass(frozen=True)
class PerGroupTimeCoeffs:
    prefill_launch: Triple
    prefill_run: Triple
    decode_launch: Triple
    decode_run: Triple


def _linear_ms(coeff: Triple, x: float, y: float) -> float:
    return coeff[0] * x + coeff[1] * y + coeff[2]


def _triple_from_yaml_list(name: str, v) -> Triple:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"pdmux fitted yaml: {name} must be a list of 3 floats, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def parse_fitted_yaml_document(doc: dict) -> Dict[Tuple[int, int], PerGroupTimeCoeffs]:
    """
    Build a map (prefill_sm, decode_sm) -> PerGroupTimeCoeffs.
    Top-level keys are ignored except for error context; each value must be a dict with
    prefill_sm, decode_sm, and the four coefficient lists.
    """
    out: Dict[Tuple[int, int], PerGroupTimeCoeffs] = {}
    if not isinstance(doc, dict):
        raise ValueError("pdmux fitted yaml: root must be a mapping")
    for _top_key, entry in doc.items():
        if not isinstance(entry, dict):
            continue
        if "prefill_sm" not in entry or "decode_sm" not in entry:
            logger.debug(
                "pdmux fitted yaml: skip entry %r (missing prefill_sm/decode_sm)",
                _top_key,
            )
            continue
        try:
            pf = int(entry["prefill_sm"])
            dd = int(entry["decode_sm"])
            key = (pf, dd)
            out[key] = PerGroupTimeCoeffs(
                prefill_launch=_triple_from_yaml_list(
                    "prefill_launch", entry["prefill_launch"]
                ),
                prefill_run=_triple_from_yaml_list("prefill_run", entry["prefill_run"]),
                decode_launch=_triple_from_yaml_list(
                    "decode_launch", entry["decode_launch"]
                ),
                decode_run=_triple_from_yaml_list("decode_run", entry["decode_run"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "pdmux fitted yaml: skip entry %r (invalid fields): %s",
                _top_key,
                e,
            )
    return out


def load_pdmux_fitted_coefficients_yaml(path: str) -> Dict[Tuple[int, int], PerGroupTimeCoeffs]:
    """Load fitted coefficients from a YAML file."""
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return parse_fitted_yaml_document(doc)


def nearest_sm_key(
    target: Tuple[int, int], keys: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """Pick key minimizing squared Euclidean distance in (prefill_sm, decode_sm)."""
    if not keys:
        raise ValueError("nearest_sm_key: empty keys")
    tp, td = target
    best = keys[0]
    best_d = (tp - best[0]) ** 2 + (td - best[1]) ** 2
    for k in keys[1:]:
        d = (tp - k[0]) ** 2 + (td - k[1]) ** 2
        if d < best_d:
            best_d = d
            best = k
    return best


def match_coefficients_for_sm_counts(
    sm_counts: List[Tuple[int, int]],
    coeff_map: Dict[Tuple[int, int], PerGroupTimeCoeffs],
    fallback: Optional[PerGroupTimeCoeffs] = None,
) -> List[PerGroupTimeCoeffs]:
    """
    For each runtime stream group g, set coeffs from coeff_map using sm_counts[g].
    Exact (prefill_sm, decode_sm) match; else nearest neighbor in YAML.
    If coeff_map is empty and fallback is set, use fallback for every group.
    """
    if not sm_counts:
        return []
    if not coeff_map:
        if fallback is None:
            raise ValueError(
                "match_coefficients_for_sm_counts: empty coeff_map and no fallback"
            )
        return [fallback] * len(sm_counts)

    keys = list(coeff_map.keys())
    out: List[PerGroupTimeCoeffs] = []
    for pf, dd in sm_counts:
        target = (int(pf), int(dd))
        if target in coeff_map:
            out.append(coeff_map[target])
            continue
        nk = nearest_sm_key(target, keys)
        logger.info(
            "pdmux fitted yaml: no exact match for (prefill_sm=%s, decode_sm=%s); "
            "using nearest (%s, %s)",
            pf,
            dd,
            nk[0],
            nk[1],
        )
        out.append(coeff_map[nk])
    return out


def default_coefficients_for_num_groups(num_groups: int) -> List[PerGroupTimeCoeffs]:
    """
    Placeholder coeffs: larger g uses smaller effective decode time (1/(g+1) scale)
    so that under a loose budget, choose_stream_group_decode_slo tends to pick
    smaller decode_sm when that group's prediction still fits — offline-fitted
    tables should replace this.
    """
    out: List[PerGroupTimeCoeffs] = []
    for g in range(num_groups):
        inv = 1.0 / float(g + 1)
        # prefill: weak dependence on placeholder
        pf = (1e-6 * inv, 1e-6 * inv, 0.5)
        dl = (0.02 * inv, 0.03 * inv, 0.15 * inv)
        dr = (0.01 * inv, 0.05 * inv, 0.2 * inv)
        out.append(
            PerGroupTimeCoeffs(
                prefill_launch=pf,
                prefill_run=pf,
                decode_launch=dl,
                decode_run=dr,
            )
        )
    return out


def decode_lb_from_schedule_batch(
    batch: "ScheduleBatch",
) -> Optional[Tuple[float, float]]:
    """Return (L, B) for decode time model, or None if global_num_tokens missing."""
    if batch.global_num_tokens is None or len(batch.global_num_tokens) == 0:
        return None
    B = float(sum(int(x) for x in batch.global_num_tokens))
    gss = getattr(batch, "global_seq_lens_sum_per_dp", None)
    if gss is not None and len(gss) > 0:
        L = float(max(int(x) for x in gss))
    elif batch.seq_lens_cpu is not None and batch.seq_lens_cpu.numel() > 0:
        L = float(batch.seq_lens_cpu.sum().item())
        logger.debug(
            "pdmux_time_model: global_seq_lens_sum_per_dp missing; using sum(seq_lens_cpu) for L"
        )
    else:
        L = 0.0
    return L, B


def prefill_fg_from_schedule_batch(batch: "ScheduleBatch") -> Tuple[float, float]:
    """Return (F, G) = (sum n_i^2, sum n_i) with n_i = extend_input_len."""
    reqs = batch.reqs
    if not reqs:
        return 0.0, 0.0
    F = 0.0
    G = 0.0
    for req in reqs:
        n = float(req.extend_input_len)
        F += n * n
        G += n
    return F, G


class PDMuxTimePredictor:
    """Predict prefill/decode times and choose stream group under decode run budget."""

    def __init__(
        self,
        per_group: List[PerGroupTimeCoeffs],
        budget_ms: float,
    ):
        self._per_group = per_group
        self.budget_ms = float(budget_ms)

    def coeffs_for_group(self, g: int) -> PerGroupTimeCoeffs:
        if g < 0 or g >= len(self._per_group):
            g = max(0, min(g, len(self._per_group) - 1))
        return self._per_group[g]

    def predict_prefill_launch_ms(self, g: int, F: float, G: float) -> float:
        c = self.coeffs_for_group(g)
        return _linear_ms(c.prefill_launch, F, G)

    def predict_prefill_run_ms(self, g: int, F: float, G: float) -> float:
        c = self.coeffs_for_group(g)
        return _linear_ms(c.prefill_run, F, G)

    def predict_decode_launch_ms(self, g: int, L: float, B: float) -> float:
        c = self.coeffs_for_group(g)
        return _linear_ms(c.decode_launch, L, B)

    def predict_decode_run_ms(self, g: int, L: float, B: float) -> float:
        c = self.coeffs_for_group(g)
        return _linear_ms(c.decode_run, L, B)

    def choose_stream_group_decode_slo(
        self,
        L: float,
        B: float,
        sm_counts: List[Tuple[int, int]],
        diag_log: bool = False,
    ) -> Optional[int]:
        """
        Among groups g with predict_decode_run_ms(g,L,B) <= budget_ms, pick minimum
        decode_sm (sm_counts[g][1]), tie-break by smallest g.

        Index g=0 is **not** a manual_divisions row: ``initialize_stream_groups`` prepends
        ``(total_sm, 0)`` (all SMs for prefill, decode_sm=0). That endpoint would always win
        the min-decode_sm tie-break and is outside the YAML multiplex grid, so it is skipped
        here. Valid SLO choices start at g=1 (first ``manual_divisions`` entry when using
        YAML). The trailing full-decode group (prefill_sm=0) remains eligible if feasible.

        Returns None if no group is feasible or coeffs shorter than sm_counts (after clamp).
        """
        n = len(sm_counts)
        if n == 0:
            return None
        per_g_lines: List[str] = []
        candidates: List[Tuple[int, int, float]] = []
        for g in range(n-1):
            # skip the first sg and the last sg, we not use predict
            if g == 0:
                if diag_log:
                    pf, dd = sm_counts[g][0], sm_counts[g][1]
                    per_g_lines.append(
                        f"  g={g} prefill_sm={pf} decode_sm={dd}: "
                        f"SKIP (full-prefill endpoint, not in manual_divisions / multiplex grid)"
                    )
                continue
            if g >= len(self._per_group):
                if diag_log:
                    per_g_lines.append(
                        f"  g={g} prefill_sm={sm_counts[g][0]} decode_sm={sm_counts[g][1]}: "
                        f"SKIP (no coeffs)"
                    )
                continue
            t_run = self.predict_decode_run_ms(g, L, B)
            ok = t_run <= self.budget_ms
            if diag_log:
                per_g_lines.append(
                    f"  g={g} prefill_sm={sm_counts[g][0]} decode_sm={sm_counts[g][1]}: "
                    f"t_decode_run_ms={t_run:.6f} budget_ms={self.budget_ms} feasible={ok}"
                )
            if ok:
                decode_sm = int(sm_counts[g][1])
                candidates.append((decode_sm, g, t_run))
        if diag_log:
            logger.info(
                "pdmux_time_predictor decode_slo detail: L=%s B=%s\n%s",
                L,
                B,
                "\n".join(per_g_lines),
            )
        if not candidates or len(candidates) == 0:
            # TODO(lbz): if no candidates, we should return the last sg
            return len(self._per_group) - 1
        candidates.sort(key=lambda x: (x[0], x[1]))
        chosen = candidates[0][1]
        logger.debug(
            "pdmux_time_model: choose_stream_group_decode_slo L=%s B=%s budget_ms=%s -> g=%s",
            L,
            B,
            self.budget_ms,
            chosen,
        )
        return chosen

    def suggest_forward_count_from_decode_prefill_ratio(
        self,
        g: int,
        F: float,
        G: float,
        decode_run_pred_ms: Optional[float],
        num_hidden_layers: int,
        split_index: int,
    ) -> Optional[int]:
        """
        Align split-prefill step size with predicted decode run time:
        forward_count ≈ round(decode_run_pred * num_hidden_layers / prefill_launch_pred),
        clamped to [1, num_hidden_layers - split_index].

        Uses the same linear prefill_launch model as elsewhere; requires a positive
        decode_run_pred_ms from the current decode batch (after MLP sync).
        """
        if decode_run_pred_ms is None or decode_run_pred_ms <= 0:
            return None
        pl = self.predict_prefill_launch_ms(g, F, G)
        if pl <= 1e-12:
            return None
        layers = float(num_hidden_layers)
        raw = decode_run_pred_ms * layers / pl
        remaining = num_hidden_layers - split_index
        if remaining <= 0:
            return None
        fc = int(round(raw))
        return max(1, min(fc, remaining))
