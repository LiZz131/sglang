#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _build_sampling_params(args: argparse.Namespace) -> Dict[str, Any]:
    sp: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    # Keep deterministic by default; allow user override
    if args.top_k is not None:
        sp["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        sp["repetition_penalty"] = args.repetition_penalty
    return sp


def _normalize_request(item: Any) -> Tuple[str, int, int]:
    """
    bench_serving.get_dataset returns "input_requests" whose exact schema depends on dataset.
    For correctness we only need a prompt text plus (prompt_len, output_len) metadata if present.

    We try to extract:
    - prompt: str
    - prompt_len: int (best-effort)
    - output_len: int (best-effort)
    """
    prompt = None
    prompt_len = 0
    output_len = 0

    # bench_serving returns DatasetRow dataclass for most datasets.
    # We avoid importing it directly here; just use attribute introspection.
    if hasattr(item, "prompt") and hasattr(item, "prompt_len"):
        try:
            prompt = getattr(item, "prompt")
            prompt_len = int(getattr(item, "prompt_len", 0) or 0)
            output_len = int(getattr(item, "output_len", 0) or 0)
        except Exception:
            prompt = None

    if isinstance(item, dict):
        # sharegpt/custom/random typically include "prompt" or "text"
        prompt = item.get("prompt", None)
        if prompt is None:
            prompt = item.get("text", None)
        prompt_len = int(item.get("prompt_len", 0) or 0)
        output_len = int(item.get("output_len", 0) or 0)
    elif isinstance(item, (list, tuple)) and len(item) >= 1:
        # Some internal variants may return tuples like (prompt, prompt_len, output_len, ...)
        prompt = item[0]
        if len(item) >= 2:
            try:
                prompt_len = int(item[1])
            except Exception:
                prompt_len = 0
        if len(item) >= 3:
            try:
                output_len = int(item[2])
            except Exception:
                output_len = 0

    if not isinstance(prompt, str):
        raise ValueError(f"Unsupported dataset item schema for prompt extraction: {type(item)}")
    return prompt, prompt_len, output_len


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True, help="sharegpt | random | custom")
    p.add_argument(
        "--dataset-path",
        default="",
        help=(
            "Dataset file path. For sharegpt: can be empty to auto-download via HF. "
            "For custom: must be a local jsonl. For random: optional (used only if you want to sample from an existing dataset)."
        ),
    )
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--output", required=True, help="Output jsonl path")

    # Sampling params for /generate
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--return-logprob", action="store_true")

    # Random dataset args (match bench_serving defaults loosely)
    p.add_argument("--random-input-len", type=int, default=256)
    p.add_argument("--random-output-len", type=int, default=64)
    p.add_argument("--random-range-ratio", type=float, default=1.0)
    p.add_argument("--model-id", default="", help="Tokenizer/model id for dataset sampling")
    p.add_argument("--prompt-suffix", default="", help="Passed to sharegpt/custom sampler")
    p.add_argument("--apply-chat-template", action="store_true")
    args = p.parse_args()

    # Reuse bench_serving dataset sampling to ensure prompt distribution matches your perf tests.
    from sglang.bench_serving import get_dataset, get_tokenizer

    class _Args:
        pass

    a = _Args()
    a.dataset_name = args.dataset_name
    a.dataset_path = args.dataset_path
    a.num_prompts = args.num_prompts
    a.prompt_suffix = args.prompt_suffix
    a.apply_chat_template = args.apply_chat_template
    a.tokenize_prompt = False

    # sharegpt/custom fields
    a.sharegpt_output_len = args.random_output_len
    a.sharegpt_context_len = args.random_input_len

    # random fields
    a.random_input_len = args.random_input_len
    a.random_output_len = args.random_output_len
    a.random_range_ratio = args.random_range_ratio

    # unused for our subset but required by get_dataset branches
    a.backend = "sglang"
    a.image_count = 0
    a.image_content = ""
    a.image_format = ""
    a.image_resolution = ""
    a.random_image_count = False
    a.gsp_num_groups = 0
    a.gsp_prompts_per_group = 0
    a.gsp_system_prompt_len = 0
    a.gsp_question_len = 0
    a.gsp_output_len = 0
    a.mooncake_workload = ""

    tokenizer = get_tokenizer(args.model_id or "meta-llama/Meta-Llama-3-8B-Instruct")
    input_requests = get_dataset(a, tokenizer, model_id=args.model_id or None)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampling_params = _build_sampling_params(args)

    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(input_requests):
            prompt, prompt_len, _ = _normalize_request(item)
            rec = {
                "id": f"{i:06d}_{uuid.uuid4().hex[:8]}",
                "text": prompt,
                "prompt_len": prompt_len,
                "sampling_params": sampling_params,
                "return_logprob": bool(args.return_logprob),
                "meta": {"dataset_name": args.dataset_name},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

