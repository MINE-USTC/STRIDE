"""
Tokenize supervised fine-tuning jsonl into a HuggingFace ``Dataset`` on disk.

Each input line must be a JSON object with keys ``instruction``, ``input``, and
``output`` (already filtered / formatted for the target module). This module does
not build or filter those rows from pipeline logs—that happens outside this package tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jsonlines
from datasets import Dataset
from transformers import AutoTokenizer


def process_example(example: dict, tokenizer, max_length: int) -> dict:
    instruction = tokenizer(
        f"<s><|im_start|>system\n{example['instruction']}<|redacted_im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|redacted_im_end|>\n"
        f"<|im_start|>assistant\n<redacted_thinking>\n\n</redacted_thinking>\n\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > max_length:
        print(f"Truncating from {len(input_ids)} to {max_length} tokens (example not trimmed in code).")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Preprocess ft_data jsonl -> Dataset save_to_disk")
    p.add_argument("--input_jsonl", required=True, type=Path)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--max_length", type=int, default=2048)
    args = p.parse_args(argv)

    out = args.output_dir or Path(str(args.input_jsonl).replace(".jsonl", "_processed"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    rows = []
    with jsonlines.open(args.input_jsonl) as f:
        for obj in f:
            rows.append(process_example(obj, tokenizer, args.max_length))
    ds = Dataset.from_list(rows)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"Saved {len(rows)} examples to {out}")


if __name__ == "__main__":
    main()
