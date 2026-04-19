"""CLI: evaluate STRIDE jsonl (optional merge with fallback_qa output)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metrics import evaluate_file


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Exact match, F1, precision, and recall on prediction jsonl")
    p.add_argument(
        "jsonl",
        type=Path,
        help="Main run jsonl (e.g. under output/<run_name>/...)",
    )
    p.add_argument(
        "--fallback-jsonl",
        type=Path,
        default=None,
        help="Optional fallback_qa jsonl merged by id when main answer is empty",
    )
    p.add_argument("--json-out", type=Path, default=None, help="Write metrics JSON")
    args = p.parse_args(argv)

    m = evaluate_file(args.jsonl, args.fallback_jsonl)
    if "error" in m:
        print(m, file=sys.stderr)
        sys.exit(1)
    lines = [
        f"n: {int(m['n'])}",
        f"EM: {m['em']:.4f}",
        f"F1 / P / R: {m['f1']:.4f} / {m['precision']:.4f} / {m['recall']:.4f}",
        f"parse_errors (get_answer): {int(m.get('parse_errors', 0))}",
    ]
    print("\n".join(lines))
    if args.json_out:
        args.json_out.write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
