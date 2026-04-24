#!/usr/bin/env python3
"""
Run requests to the endpoint.

Usage:
    python run_requests.py --requests requests.jsonl --output results_dp_attn_no_logp.jsonl --host 127.0.0.1 --port 31001 --path /generate
    python run_requests.py --requests requests.jsonl --output results_normal_no_logp.jsonl --host 127.0.0.1 --port 30001 --path /generate
    python run_requests.py --requests requests.jsonl --output results.jsonl --url http://127.0.0.1:30001/generate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import aiohttp


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _now_s() -> float:
    return time.perf_counter()


def _endpoint_url(args: argparse.Namespace) -> str:
    if args.url is not None:
        return args.url
    path = args.path if args.path.startswith("/") else f"/{args.path}"
    return f"http://{args.host}:{args.port}{path}"


async def _post_one(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    st = _now_s()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            latency_s = _now_s() - st
            ok = resp.status == 200
            text = await resp.text()
            try:
                resp_json = json.loads(text)
            except Exception:
                resp_json = {"_raw_text": text}
            return {
                "ok": ok,
                "status_code": resp.status,
                "latency_s": latency_s,
                "response": resp_json,
            }
    except Exception as e:
        latency_s = _now_s() - st
        return {
            "ok": False,
            "status_code": None,
            "latency_s": latency_s,
            "response": None,
            "error": repr(e),
        }


async def _producer(
    queue: asyncio.Queue,
    reqs: Iterable[Dict[str, Any]],
    request_rate: Optional[float],
    num_consumers: int,
) -> None:
    """
    Put requests into queue.
    If request_rate is set, the *start times* of requests are rate-limited (Poisson-like uniform spacing).
    """
    if request_rate is not None and request_rate <= 0:
        request_rate = None

    next_t = _now_s()
    interval = (1.0 / request_rate) if request_rate else 0.0

    for rec in reqs:
        if request_rate:
            now = _now_s()
            if now < next_t:
                await asyncio.sleep(next_t - now)
            next_t = max(next_t + interval, _now_s())
        await queue.put(rec)

    # One sentinel per consumer for clean shutdown.
    for _ in range(max(int(num_consumers), 1)):
        await queue.put(None)


async def _consumer(
    name: str,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: float,
    out_path: Path,
    file_lock: asyncio.Lock,
    progress: Dict[str, Any],
    progress_lock: asyncio.Lock,
) -> None:
    while True:
        rec = await queue.get()
        try:
            if rec is None:
                return

            rid = rec["id"]
            payload = {
                "text": rec["text"],
                "sampling_params": rec.get("sampling_params", {}),
                "stream": False,
            }
            if "return_logprob" in rec:
                payload["return_logprob"] = bool(rec["return_logprob"])

            result = await _post_one(session, url=url, payload=payload, timeout_s=timeout_s)
            row = {
                "id": rid,
                **result,
                "request": payload,
                "worker": name,
            }
            line = json.dumps(row, ensure_ascii=False) + "\n"

            async with file_lock:
                with out_path.open("a", encoding="utf-8") as out:
                    out.write(line)

            if progress.get("enabled", False):
                now = _now_s()
                async with progress_lock:
                    progress["done"] += 1
                    done = progress["done"]
                    total = progress["total"]
                    last_print_t = progress["last_print_t"]
                    if now - last_print_t >= progress["print_every_s"]:
                        elapsed = max(now - progress["start_t"], 1e-9)
                        qps = done / elapsed
                        avg_lat = progress["lat_sum_s"] / max(done, 1)
                        ok = progress["ok"]
                        fail = progress["fail"]
                        print(
                            f"[run_requests] {done}/{total} done | "
                            f"ok={ok} fail={fail} | qps={qps:.2f} | avg_latency_s={avg_lat:.3f}",
                            flush=True,
                        )
                        progress["last_print_t"] = now

            # Update stats (keep lock scope small)
            if progress.get("enabled", False):
                async with progress_lock:
                    progress["lat_sum_s"] += float(result.get("latency_s", 0.0) or 0.0)
                    if result.get("ok", False):
                        progress["ok"] += 1
                    else:
                        progress["fail"] += 1
        finally:
            queue.task_done()


async def _run_async(args: argparse.Namespace) -> None:
    req_path = Path(args.requests)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.append:
        out_path.unlink()

    reqs = list(_iter_jsonl(req_path))
    url = _endpoint_url(args)

    queue: asyncio.Queue = asyncio.Queue(maxsize=max(args.concurrency * 2, 1))
    file_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    progress: Dict[str, Any] = {
        "enabled": bool(args.progress),
        "total": len(reqs),
        "done": 0,
        "ok": 0,
        "fail": 0,
        "lat_sum_s": 0.0,
        "start_t": _now_s(),
        "last_print_t": _now_s(),
        "print_every_s": float(args.progress_every),
    }

    async with aiohttp.ClientSession() as session:
        consumers = [
            asyncio.create_task(
                _consumer(
                    name=f"w{i}",
                    queue=queue,
                    session=session,
                    url=url,
                    timeout_s=args.timeout,
                    out_path=out_path,
                    file_lock=file_lock,
                    progress=progress,
                    progress_lock=progress_lock,
                )
            )
            for i in range(args.concurrency)
        ]
        prod = asyncio.create_task(_producer(queue, reqs, args.request_rate, args.concurrency))

        await prod
        await queue.join()
        # At this point all tasks are done; consumers should have received their sentinels and exited.
        await asyncio.gather(*consumers, return_exceptions=True)

    if progress.get("enabled", False):
        elapsed = max(_now_s() - progress["start_t"], 1e-9)
        done = progress["done"]
        qps = done / elapsed
        avg_lat = progress["lat_sum_s"] / max(done, 1)
        print(
            f"[run_requests] finished {done}/{progress['total']} | "
            f"ok={progress['ok']} fail={progress['fail']} | qps={qps:.2f} | avg_latency_s={avg_lat:.3f}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requests", required=True, help="requests.jsonl produced by prepare_dataset_jsonl.py")
    p.add_argument(
        "--url",
        default=None,
        help="Full endpoint URL. If unset, built from --host, --port, and --path.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Used when --url is not set.")
    p.add_argument("--port", type=int, default=30001, help="Used when --url is not set.")
    p.add_argument("--path", default="/generate", help="Path when --url is not set (leading / optional).")
    p.add_argument("--output", required=True, help="results jsonl output path")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--request-rate", type=float, default=None, help="Requests per second. If unset, run as fast as allowed by concurrency.")
    p.add_argument("--concurrency", type=int, default=1, help="Max in-flight requests.")
    p.add_argument("--append", action="store_true", help="Append to output instead of overwriting.")
    p.add_argument("--progress", action="store_true", help="Print progress periodically.")
    p.add_argument("--progress-every", type=float, default=2.0, help="Seconds between progress prints (when --progress is set).")
    args = p.parse_args()

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be >= 1")

    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
