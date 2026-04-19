"""`python -m stride` prints pointers to subcommands."""

from __future__ import annotations

USAGE = """
From the repository root after ``pip install -r requirements.txt``:

  python -m stride.pipeline
  python -m stride.eval
  python -m stride.ft_preprocess
  python -m stride.meta_planer
  python -m stride.supervisor
  python -m stride.fallback_qa
  python -m stride.lora_ft
  python -m stride.lora_dpo
  python -m stride.data_prep
  python -m stride.build_corpus_index
  python -m stride.build_ft_dataset

See README.md next to this file for paths and data setup.
"""


def main() -> None:
    print(USAGE.strip())


if __name__ == "__main__":
    main()
