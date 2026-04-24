#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _index_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        m[row["id"]] = row
    return m


def _find_token_ids(resp: Any) -> Optional[List[int]]:
    """Best-effort extraction of token ids from /generate response."""
    if not isinstance(resp, dict):
        return None
    # common fields seen in sglang variants
    for k in ("output_ids", "token_ids", "output_token_ids"):
        v = resp.get(k, None)
        if isinstance(v, list) and (not v or isinstance(v[0], int)):
            return v
    # nested under choices (openai-like)
    choices = resp.get("choices", None)
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            for k in ("output_ids", "token_ids"):
                v = c0.get(k, None)
                if isinstance(v, list) and (not v or isinstance(v[0], int)):
                    return v
    return None


def _find_finish_reason(resp: Any) -> Optional[str]:
    if not isinstance(resp, dict):
        return None
    for k in ("finish_reason", "finishReason", "stop_reason", "stopReason"):
        v = resp.get(k, None)
        if isinstance(v, str):
            return v
    choices = resp.get("choices", None)
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            v = c0.get("finish_reason", None)
            if isinstance(v, str):
                return v
    return None


def _find_next_token_logprobs(resp: Any) -> Optional[List[float]]:
    """
    Best-effort extraction for next-token logprobs.
    Depending on server settings, these may be absent.
    """
    if not isinstance(resp, dict):
        return None
    # common: logits_output.next_token_logprobs
    lo = resp.get("logits_output", None)
    if isinstance(lo, dict):
        v = lo.get("next_token_logprobs", None)
        if isinstance(v, list) and (not v or isinstance(v[0], (int, float))):
            return [float(x) for x in v]
    # openai-like: choices[0].logprobs
    choices = resp.get("choices", None)
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        lp = choices[0].get("logprobs", None)
        if isinstance(lp, dict):
            v = lp.get("token_logprobs", None)
            if isinstance(v, list) and (not v or isinstance(v[0], (int, float))):
                return [float(x) for x in v]
    return None


def _first_diff(a: List[int], b: List[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _max_abs_diff(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    m = 0.0
    for i in range(n):
        m = max(m, abs(xs[i] - ys[i]))
    # length mismatch counts as infinity-like
    if len(xs) != len(ys):
        m = max(m, float("inf"))
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="baseline results jsonl")
    p.add_argument("--b", required=True, help="workaround results jsonl")
    p.add_argument("--c", required=True, help="targetfix results jsonl")
    p.add_argument("--report", required=True, help="report output path")
    p.add_argument("--compare-logprob", action="store_true")
    p.add_argument("--logprob-atol", type=float, default=1e-4)
    args = p.parse_args()

    a = _index_by_id(Path(args.a))
    b = _index_by_id(Path(args.b))
    c = _index_by_id(Path(args.c))

    ids = sorted(set(a.keys()) & set(b.keys()) & set(c.keys()))
    missing = (set(a.keys()) | set(b.keys()) | set(c.keys())) - set(ids)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ok_cnt = 0
    fail_cnt = 0
    with report_path.open("w", encoding="utf-8") as out:
        if missing:
            out.write(f"MISSING_IDS count={len(missing)}\n")
            for mid in sorted(missing)[:50]:
                out.write(f"  - {mid}\n")
            out.write("\n")

        for rid in ids:
            ra, rb, rc = a[rid], b[rid], c[rid]
            if not (ra.get("ok") and rb.get("ok") and rc.get("ok")):
                fail_cnt += 1
                out.write(f"FAIL http ok mismatch id={rid}\n")
                out.write(f"  a.ok={ra.get('ok')} b.ok={rb.get('ok')} c.ok={rc.get('ok')}\n")
                continue

            ta = _find_token_ids(ra.get("response"))
            tb = _find_token_ids(rb.get("response"))
            tc = _find_token_ids(rc.get("response"))
            fa = _find_finish_reason(ra.get("response"))
            fb = _find_finish_reason(rb.get("response"))
            fc = _find_finish_reason(rc.get("response"))

            if ta is None or tb is None or tc is None:
                fail_cnt += 1
                out.write(f"FAIL missing token ids id={rid}\n")
                out.write(f"  token_ids: a={ta is not None} b={tb is not None} c={tc is not None}\n")
                continue

            d_ab = _first_diff(ta, tb)
            d_ac = _first_diff(ta, tc)
            if d_ab is not None or d_ac is not None:
                fail_cnt += 1
                out.write(f"FAIL token mismatch id={rid}\n")
                out.write(f"  len a={len(ta)} b={len(tb)} c={len(tc)}\n")
                out.write(f"  first_diff a~b={d_ab} a~c={d_ac}\n")
                if d_ab is not None:
                    i = d_ab
                    out.write(f"  a[{i}]={ta[i] if i < len(ta) else None} b[{i}]={tb[i] if i < len(tb) else None}\n")
                if d_ac is not None:
                    i = d_ac
                    out.write(f"  a[{i}]={ta[i] if i < len(ta) else None} c[{i}]={tc[i] if i < len(tc) else None}\n")
                continue

            if (fa, fb, fc) != (None, None, None) and not (fa == fb == fc):
                fail_cnt += 1
                out.write(f"FAIL finish_reason mismatch id={rid}\n")
                out.write(f"  a={fa} b={fb} c={fc}\n")
                continue

            if args.compare_logprob:
                la = _find_next_token_logprobs(ra.get("response"))
                lb = _find_next_token_logprobs(rb.get("response"))
                lc = _find_next_token_logprobs(rc.get("response"))
                if la is None or lb is None or lc is None:
                    fail_cnt += 1
                    out.write(f"FAIL missing logprob id={rid}\n")
                    out.write(f"  logprob: a={la is not None} b={lb is not None} c={lc is not None}\n")
                    continue
                m_ab = _max_abs_diff(la, lb)
                m_ac = _max_abs_diff(la, lc)
                if not (m_ab <= args.logprob_atol and m_ac <= args.logprob_atol):
                    fail_cnt += 1
                    out.write(f"FAIL logprob drift id={rid}\n")
                    out.write(f"  max_abs_diff a~b={m_ab} a~c={m_ac} atol={args.logprob_atol}\n")
                    continue

            ok_cnt += 1

        out.write("\n")
        out.write(f"SUMMARY ok={ok_cnt} fail={fail_cnt} total={len(ids)}\n")


if __name__ == "__main__":
    main()

