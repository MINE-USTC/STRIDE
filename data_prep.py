"""
Rebuild paper-style splits from upstream MuSiQue / HotpotQA-style jsonl:

- Subsample the training pool (e.g. 10k) with a fixed seed.
- Build an extended test file: existing dev/test slice + N extra questions sampled from
  **training** rows that are **not** in the subsampled training set (fixed split recipe).

Upstream rows are expected to include at least: question_id, question_text, answers_objects,
contexts (and optionally pinned_contexts, reasoning_steps); see ``convert_upstream_to_stride``.

STRIDE jsonl rows use: id, question, answer, contexts, reasoning_steps, corpus, supporting_count.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import jsonlines


def _row_id(obj: dict[str, Any]) -> str:
    if "id" in obj:
        return str(obj["id"])
    if "question_id" in obj:
        return str(obj["question_id"])
    raise KeyError("row has neither id nor question_id")


def load_ids(path: Path) -> set[str]:
    out: set[str] = set()
    with jsonlines.open(path) as r:
        for obj in r:
            out.add(_row_id(obj))
    return out


def convert_upstream_to_stride(obj: dict[str, Any]) -> dict[str, Any]:
    """Map upstream fields to STRIDE v1 ``test.jsonl``-style rows (id, question, answer, contexts, …)."""
    data: dict[str, Any] = {}
    data["id"] = obj["question_id"]
    data["question"] = obj["question_text"]
    if len(obj["answers_objects"]) == 1:
        data["answer"] = obj["answers_objects"][0]["spans"][0]
    else:
        print("multiple answers!!!!!!", file=sys.stderr)
    corpus: list = [[], [], [], []]
    contexts: list = []
    count = 0
    if "pinned_contexts" in obj:
        for item in obj["pinned_contexts"]:
            if item["is_supporting"] is True:
                count += 1
            item["paragraph_text"] = item["paragraph_text"].replace("\n", " ")
            contexts.append(item)
    for idx, item in enumerate(obj["contexts"]):
        if item["is_supporting"] is True:
            count += 1
        item["paragraph_text"] = item["paragraph_text"].replace("\n", " ")
        contexts.append(item)
        corpus[0].append(idx)
        corpus[1].append(item["title"])
        corpus[2].append(item["paragraph_text"])
        corpus[3].append(item["is_supporting"])
    data["contexts"] = contexts
    data["reasoning_steps"] = obj.get("reasoning_steps", [])
    data["corpus"] = corpus
    data["supporting_count"] = count
    return data


def sample_train(
    upstream_train: Path,
    output: Path,
    n: int,
    seed: int,
) -> None:
    """Write *upstream-format* jsonl: random subset of size ``n`` (or all if smaller)."""
    random.seed(seed)
    with jsonlines.open(upstream_train) as r:
        pool = [obj for obj in r]
    k = min(n, len(pool))
    sample = random.sample(pool, k)
    output.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output, "w") as w:
        for obj in sample:
            w.write(obj)
    print(f"Wrote {len(sample)} / {len(pool)} rows to {output}")


def merge_test_with_train_extras(
    *,
    base_test_stride: Path,
    upstream_train_full: Path,
    train_sample_upstream: Path,
    output: Path,
    extra_n: int,
    seed: int,
) -> dict[str, Any]:
    """
    Append ``extra_n`` questions from full train, excluding ids present in ``train_sample_upstream``,
    then prepend/append to ``base_test_stride`` (STRIDE-format jsonl).

    Returns a report dict with overlap checks.
    """
    random.seed(seed)
    excluded = load_ids(train_sample_upstream)
    base_rows: list[dict[str, Any]] = []
    with jsonlines.open(base_test_stride) as r:
        for obj in r:
            base_rows.append(obj)
    base_ids = {str(o["id"]) for o in base_rows}

    pool: list[dict[str, Any]] = []
    with jsonlines.open(upstream_train_full) as r:
        for obj in r:
            qid = str(obj["question_id"])
            if qid in excluded:
                continue
            pool.append(obj)

    k = min(extra_n, len(pool))
    if k < extra_n:
        print(
            f"Warning: only {len(pool)} train rows outside the {len(excluded)}-id exclusion set; "
            f"sampling {k} instead of {extra_n}.",
            file=sys.stderr,
        )
    extras = random.sample(pool, k) if k else []
    extra_stride = [convert_upstream_to_stride(obj) for obj in extras]
    extra_ids = {str(o["id"]) for o in extra_stride}

    report = {
        "base_test_count": len(base_rows),
        "extra_count": len(extra_stride),
        "overlap_extra_vs_train_sample": sorted(extra_ids & excluded),
        "overlap_extra_vs_base_test": sorted(extra_ids & base_ids),
        "overlap_base_vs_train_sample": sorted(base_ids & excluded),
    }

    merged = base_rows + extra_stride
    output.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output, "w") as w:
        for obj in merged:
            w.write(obj)

    print(json.dumps(report, indent=2))
    print(f"Wrote {len(merged)} rows to {output}")
    return report


def check_overlap(paths: list[Path], id_keys: tuple[str, ...] = ("id", "question_id")) -> None:
    """Print pairwise intersection sizes for question ids across jsonl files."""

    def ids_in(p: Path) -> set[str]:
        s: set[str] = set()
        with jsonlines.open(p) as r:
            for obj in r:
                for k in id_keys:
                    if k in obj:
                        s.add(str(obj[k]))
                        break
                else:
                    raise KeyError(f"{p}: no id key in {id_keys}")
        return s

    sets = [(str(p), ids_in(p)) for p in paths]
    for i, (ni, si) in enumerate(sets):
        for j, (nj, sj) in enumerate(sets):
            if j <= i:
                continue
            inter = si & sj
            print(f"{ni} ∩ {nj}: {len(inter)} ids")
            if len(inter) and len(inter) <= 20:
                print(f"  {sorted(inter)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="STRIDE data prep: train subsample + extended test")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("sample_train", help="Subsample upstream train jsonl")
    p_train.add_argument("--upstream_train", type=Path, required=True)
    p_train.add_argument("--output", type=Path, required=True)
    p_train.add_argument("--n", type=int, default=10_000)
    p_train.add_argument("--seed", type=int, default=42)

    p_merge = sub.add_parser(
        "merge_test",
        help="Base STRIDE test + N extras from train (excluding train subsample ids)",
    )
    p_merge.add_argument("--base_test", type=Path, required=True, help="STRIDE-format test jsonl (e.g. 500 rows)")
    p_merge.add_argument("--upstream_train", type=Path, required=True, help="Full upstream train jsonl")
    p_merge.add_argument(
        "--train_sample",
        type=Path,
        required=True,
        help="Upstream-format jsonl of subsampled train (ids to exclude from extra pool)",
    )
    p_merge.add_argument("--output", type=Path, required=True)
    p_merge.add_argument("--extra_n", type=int, default=500)
    p_merge.add_argument("--seed", type=int, default=42)

    p_ov = sub.add_parser("check_overlap", help="Pairwise id intersection across jsonl files")
    p_ov.add_argument("jsonl", type=Path, nargs="+")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "sample_train":
        sample_train(args.upstream_train, args.output, args.n, args.seed)
    elif args.cmd == "merge_test":
        merge_test_with_train_extras(
            base_test_stride=args.base_test,
            upstream_train_full=args.upstream_train,
            train_sample_upstream=args.train_sample,
            output=args.output,
            extra_n=args.extra_n,
            seed=args.seed,
        )
    elif args.cmd == "check_overlap":
        check_overlap(list(args.jsonl))


if __name__ == "__main__":
    main()
