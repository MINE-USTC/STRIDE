"""
Build a FAISS index on disk in the same layout ``DenseRetriever.load_index`` expects
(``faiss.index`` + ``document.vecstore.npz`` with ``documents``, ``titles``, ``embeddings``).

Typical source: STRIDE-format jsonl where each line has a ``contexts`` list of
``{title, paragraph_text}`` objects (same fields the Supervisor retrieves over).

Usage (from the directory that contains ``pipeline.py``, with its parent on
``PYTHONPATH``)::

    python -m stride.build_corpus_index \\
        --input_jsonl /path/to/train_or_dev.jsonl \\
        --output_dir faiss_index/hotpotqa/index \\
        --format stride_contexts
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Iterator

import jsonlines

from stride.contriever_model import load_contriever_and_tokenizer
from stride.my_retriever import DenseRetriever


def _fingerprint(title: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(title.encode("utf-8", errors="replace"))
    h.update(b"\0")
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()


def iter_docs_stride_contexts(
    input_jsonl: Path,
    *,
    dedupe: bool,
) -> Iterator[tuple[str, str]]:
    """Yield (title, paragraph_text) from STRIDE rows."""
    seen: set[str] = set()
    with jsonlines.open(input_jsonl) as reader:
        for obj in reader:
            blocks: list[dict] = []
            blocks.extend(obj.get("pinned_contexts") or [])
            blocks.extend(obj.get("contexts") or [])
            for ctx in blocks:
                title = str(ctx.get("title", "") or "")
                text = str(ctx.get("paragraph_text", ctx.get("text", "")) or "")
                if not text.strip():
                    continue
                key = _fingerprint(title, text)
                if dedupe and key in seen:
                    continue
                seen.add(key)
                yield title, text


def iter_docs_records(
    input_jsonl: Path,
    *,
    dedupe: bool,
) -> Iterator[tuple[str, str]]:
    """Yield (title, text) from generic jsonl: each object has ``title`` and ``text``."""
    seen: set[str] = set()
    with jsonlines.open(input_jsonl) as reader:
        for obj in reader:
            title = str(obj.get("title", "") or "")
            text = str(obj.get("text", obj.get("paragraph_text", "")) or "")
            if not text.strip():
                continue
            key = _fingerprint(title, text)
            if dedupe and key in seen:
                continue
            seen.add(key)
            yield title, text


def batched(pairs: Iterable[tuple[str, str]], batch_size: int) -> Iterator[tuple[list[str], list[str]]]:
    titles: list[str] = []
    texts: list[str] = []
    for title, text in pairs:
        titles.append(title)
        texts.append(text)
        if len(texts) >= batch_size:
            yield titles, texts
            titles, texts = [], []
    if texts:
        yield titles, texts


def build_index(
    *,
    input_jsonl: Path,
    output_dir: Path,
    retriever_model_path: str,
    batch_size: int,
    fmt: str,
    dedupe: bool,
) -> None:
    if fmt == "stride_contexts":
        pair_iter = iter_docs_stride_contexts(input_jsonl, dedupe=dedupe)
    elif fmt == "records":
        pair_iter = iter_docs_records(input_jsonl, dedupe=dedupe)
    else:
        raise ValueError(f"unknown format: {fmt}")

    model, tok = load_contriever_and_tokenizer(retriever_model_path)
    r = DenseRetriever(model, tok, batch_size=batch_size)

    n = 0
    for titles, texts in batched(pair_iter, batch_size):
        r.add_docs(texts, titles)
        n += len(texts)
        if n and n % (batch_size * 20) == 0:
            print(f"indexed {n} passages ...")

    if r.ctr == 0:
        raise SystemExit("No passages indexed; check --format and jsonl fields.")

    output_dir = output_dir.expanduser().resolve()
    r.save_index(str(output_dir))
    print(f"Done. Wrote {r.ctr} vectors to {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build FAISS + vecstore index compatible with stride.supervisor / fallback_qa.",
    )
    p.add_argument("--input_jsonl", type=Path, required=True, help="Source jsonl")
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to create (faiss.index + document.vecstore.npz)",
    )
    p.add_argument(
        "--retriever_model_path",
        default="facebook/contriever",
        help="Must match inference --retriever_model_path",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--format",
        choices=("stride_contexts", "records"),
        default="stride_contexts",
        help="stride_contexts: STRIDE rows with contexts[]. records: {title,text} per line",
    )
    p.add_argument(
        "--no_dedupe",
        action="store_true",
        help="Keep duplicate (title, text) passages across rows",
    )
    args = p.parse_args()
    build_index(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        retriever_model_path=args.retriever_model_path,
        batch_size=args.batch_size,
        fmt=args.format,
        dedupe=not args.no_dedupe,
    )


if __name__ == "__main__":
    main()
