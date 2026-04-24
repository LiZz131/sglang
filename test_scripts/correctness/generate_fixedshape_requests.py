#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _is_printable_text(s: str) -> bool:
    if not s:
        return False
    # Avoid multiline / control-heavy strings that make logs painful.
    if "\n" in s or "\r" in s or "\t" in s:
        return False
    printable = sum(ch.isprintable() for ch in s)
    return printable / max(len(s), 1) >= 0.98


def _rand_id(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice(string.hexdigits.lower()) for _ in range(n))


def _build_token_pool(tokenizer, rng: random.Random, pool_size: int) -> List[int]:
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        vocab_size = int(getattr(tokenizer, "__len__", lambda: 0)())
    if vocab_size <= 0:
        raise RuntimeError("Cannot determine tokenizer vocab_size")

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    pool: List[int] = []
    attempts = 0
    # Keep this bounded so we fail fast on weird tokenizers.
    max_attempts = max(20000, pool_size * 200)
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        tid = rng.randrange(vocab_size)
        if tid in special_ids:
            continue
        txt = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not _is_printable_text(txt):
            continue
        pool.append(tid)

    if len(pool) < max(32, pool_size // 4):
        raise RuntimeError(
            f"Failed to build a usable token pool (got {len(pool)}/{pool_size}). "
            "Try a different tokenizer/model, or lower --pool-size."
        )
    return pool


def _make_prompt_tokens(pool: Sequence[int], prompt_tokens: int, rng: random.Random) -> List[int]:
    return [pool[rng.randrange(len(pool))] for _ in range(prompt_tokens)]

def _parse_prompt_tokens_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate a JSONL requests file whose prompts have a fixed token length. "
            "Prompts are randomized at token-level to reduce KV-cache duplication."
        )
    )
    p.add_argument("--model-path", required=True, help="Tokenizer/model path (HF repo or local dir).")
    p.add_argument("--output", required=True, help="Output requests.jsonl path.")
    p.add_argument("--num-requests", type=int, default=64, help="Number of requests to generate (single-shape mode).")
    p.add_argument("--prompt-tokens", type=int, default=None, help="Exact prompt token length (single-shape mode).")
    p.add_argument(
        "--prompt-tokens-list",
        default=None,
        help="Multiple prompt token lengths, e.g. '1,2,4,8,16,32,64 96 128'.",
    )
    p.add_argument(
        "--per-shape",
        type=int,
        default=2,
        help="Requests per shape when --prompt-tokens-list is set.",
    )
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=512, help="How many token ids to sample as building blocks.")
    p.add_argument("--return-logprob", action="store_true", help="Set return_logprob=true in requests.")
    args = p.parse_args()

    if args.prompt_tokens_list:
        if args.per_shape <= 0:
            raise SystemExit("--per-shape must be >= 1")
        prompt_tokens_list = _parse_prompt_tokens_list(args.prompt_tokens_list)
        if not prompt_tokens_list:
            raise SystemExit("--prompt-tokens-list must not be empty")
        if any(x <= 0 for x in prompt_tokens_list):
            raise SystemExit("--prompt-tokens-list values must be >= 1")
        total_requests = len(prompt_tokens_list) * int(args.per_shape)
    else:
        if args.num_requests <= 0:
            raise SystemExit("--num-requests must be >= 1")
        if args.prompt_tokens is None or args.prompt_tokens <= 0:
            raise SystemExit("--prompt-tokens must be >= 1 (single-shape mode)")
        prompt_tokens_list = [int(args.prompt_tokens)]
        total_requests = int(args.num_requests)

    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be >= 1")

    # Keep tokenizer side effects (parallelism) predictable for profiling.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    rng = random.Random(args.seed)

    pool = _build_token_pool(tokenizer, rng=rng, pool_size=args.pool_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sampling_params: Dict[str, Any] = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
    }

    with out_path.open("w", encoding="utf-8") as f:
        i = 0
        for prompt_tokens in prompt_tokens_list:
            reps = int(args.per_shape) if args.prompt_tokens_list else total_requests
            for _rep in range(reps):
                toks = _make_prompt_tokens(pool, prompt_tokens=prompt_tokens, rng=rng)
                text = tokenizer.decode(toks, clean_up_tokenization_spaces=False)
                # Defensive: ensure decode didn't introduce newlines etc.
                if not _is_printable_text(text):
                    for _ in range(50):
                        toks = _make_prompt_tokens(pool, prompt_tokens=prompt_tokens, rng=rng)
                        text = tokenizer.decode(toks, clean_up_tokenization_spaces=False)
                        if _is_printable_text(text):
                            break
                    else:
                        raise RuntimeError(
                            "Failed to generate a printable prompt; lower --pool-size or change tokenizer."
                        )

                rec = {
                    "id": f"{i:06d}_{_rand_id(rng)}",
                    "text": text,
                    "prompt_len": int(prompt_tokens),
                    "sampling_params": sampling_params,
                    "return_logprob": bool(args.return_logprob),
                    "meta": {
                        "generator": "generate_fixedshape_requests.py",
                        "prompt_tokens": int(prompt_tokens),
                        "seed": int(args.seed),
                        "mode": "multi" if args.prompt_tokens_list else "single",
                        "total_requests": int(total_requests),
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                i += 1


if __name__ == "__main__":
    main()

