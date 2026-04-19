"""
STRIDE inference: Meta-Planner -> Supervisor (E+R) -> optional Fallback Reasoner.

Runs other entrypoints as subprocesses with cwd set to the repository root
(the directory that contains this file and ``meta_planer.py``, etc.).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from paths import default_run_name

_REPO = Path(__file__).resolve().parent
_DEFAULT_FAISS = str(Path(__file__).resolve().parent / "faiss_index" / "dataset" / "index")


def _py() -> str:
    return sys.executable


def run_meta_plan(args: argparse.Namespace) -> None:
    cmd = [
        _py(),
        "-m",
        "meta_planer",
        "--input_jsonl",
        args.input_jsonl,
        "--run_name",
        args.run_name,
        "--write_file_name",
        args.meta_write_name,
        "--prompt_file",
        args.meta_prompt_file,
        "--batch_size",
        str(args.batch_size),
        "--model_path",
        args.model_path,
        "--run_data_num",
        str(args.run_data_num),
        "--max_model_len",
        str(args.max_model_len_meta),
        "--max_num_seqs",
        str(args.max_num_seqs),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
    ]
    if args.think_mode:
        cmd.extend(["--think_mode", "True"])
    _append_vllm_lora_meta(cmd, args)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print("[stride] Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_REPO), env=env)


def run_supervisor(args: argparse.Namespace, plan_file_name: str) -> None:
    cmd = [
        _py(),
        "-m",
        "supervisor",
        "--input_jsonl",
        args.input_jsonl,
        "--run_name",
        args.run_name,
        "--plan_file_name",
        plan_file_name,
        "--write_file_name",
        args.supervisor_write_name,
        "--s_prompt_file",
        args.s_prompt_file,
        "--e_prompt_file",
        args.e_prompt_file,
        "--r_prompt_file",
        args.r_prompt_file,
        "--top_k_docs",
        str(args.top_k_docs),
        "--max_iteration",
        str(args.max_iteration),
        "--failed_threshold",
        str(args.failed_threshold),
        "--bs_per_iter",
        str(args.bs_per_iter),
        "--batch_size",
        str(args.batch_size),
        "--model_path",
        args.model_path,
        "--run_data_num",
        str(args.run_data_num),
        "--retriever_model_path",
        args.retriever_model_path,
        "--faiss_index_path",
        args.faiss_index_path,
        "--max_model_len",
        str(args.max_model_len_super),
        "--max_num_seqs",
        str(args.max_num_seqs),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
    ]
    if args.index_corpus:
        cmd.extend(["--index_corpus", args.index_corpus])
    if args.think_mode:
        cmd.extend(["--think_mode", "True"])
    _append_vllm_lora_supervisor(cmd, args)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print("[stride] Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_REPO), env=env)


def run_fallback(args: argparse.Namespace, used_result_file: str, plan_file_name: str) -> None:
    cmd = [
        _py(),
        "-m",
        "fallback_qa",
        "--run_name",
        args.run_name,
        "--used_result_file",
        used_result_file,
        "--plan_file_name",
        plan_file_name,
        "--write_file_name",
        args.fallback_write_name,
        "--batch_size",
        str(args.batch_size),
        "--model_path",
        args.model_path,
        "--top_k_docs",
        str(args.top_k_docs_fallback),
        "--retriever_model_path",
        args.retriever_model_path,
        "--faiss_index_path",
        args.faiss_index_path,
        "--max_model_len",
        str(args.max_model_len_fallback),
        "--max_num_seqs",
        str(args.max_num_seqs),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
    ]
    if args.index_corpus:
        cmd.extend(["--index_corpus", args.index_corpus])
    if args.input_jsonl:
        cmd.extend(["--input_jsonl", args.input_jsonl])
    if args.think_mode:
        cmd.extend(["--think_mode", "True"])
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print("[stride] Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_REPO), env=env)


def _append_vllm_lora_meta(cmd: list[str], args: argparse.Namespace) -> None:
    if getattr(args, "lora_meta", None):
        cmd.extend(["--lora_meta", args.lora_meta])
        cmd.extend(["--max_lora_rank", str(args.max_lora_rank)])
        cmd.extend(["--max_loras", str(args.max_loras)])


def _append_vllm_lora_supervisor(cmd: list[str], args: argparse.Namespace) -> None:
    if getattr(args, "lora_supervisor", None):
        cmd.extend(["--lora_supervisor", args.lora_supervisor])
    if getattr(args, "lora_extractor", None):
        cmd.extend(["--lora_extractor", args.lora_extractor])
    if getattr(args, "lora_reasoner", None):
        cmd.extend(["--lora_reasoner", args.lora_reasoner])
    if any(
        [
            getattr(args, "lora_supervisor", None),
            getattr(args, "lora_extractor", None),
            getattr(args, "lora_reasoner", None),
        ]
    ):
        cmd.extend(["--max_lora_rank", str(args.max_lora_rank)])
        cmd.extend(["--max_loras", str(args.max_loras)])


def plan_file_basename(args: argparse.Namespace) -> str:
    from paths import meta_plan_relative_for_supervisor

    return meta_plan_relative_for_supervisor(args.meta_write_name)


def used_result_relpath(args: argparse.Namespace, plan_basename: str) -> str:
    from paths import supervisor_result_relpath

    return supervisor_result_relpath(
        plan_basename,
        args.supervisor_write_name,
        args.top_k_docs,
        args.max_iteration,
        args.failed_threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STRIDE: meta-plan -> supervisor -> optional fallback QA",
    )
    p.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to input jsonl (id, question, ...). Same file is used for meta-plan and supervisor.",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Output subfolder under meta_plans/ and output/ (default: stem of input_jsonl)",
    )
    p.add_argument(
        "--index_corpus",
        default=None,
        help="Substring replacing 'dataset' in --faiss_index_path (default: same as run_name)",
    )
    p.add_argument("--cuda_visible_devices", default="")
    p.add_argument("--skip_meta", action="store_true")
    p.add_argument("--skip_supervisor", action="store_true")
    p.add_argument(
        "--run_fallback",
        action="store_true",
        help="Run Fallback Reasoner on cases with no main answer",
    )

    p.add_argument("--meta_write_name", default="meta_plan")
    p.add_argument("--meta_prompt_file", default="meta_plan")

    p.add_argument("--supervisor_write_name", default="stride")
    p.add_argument("--s_prompt_file", default="default")
    p.add_argument("--e_prompt_file", default="default")
    p.add_argument("--r_prompt_file", default="default")
    p.add_argument("--top_k_docs", type=int, default=5)
    p.add_argument("--max_iteration", type=int, default=5)
    p.add_argument("--failed_threshold", type=int, default=2)
    p.add_argument("--bs_per_iter", type=int, default=8)

    p.add_argument("--fallback_write_name", default="fallback_qa")
    p.add_argument("--top_k_docs_fallback", type=int, default=5)
    p.add_argument("--max_model_len_meta", type=int, default=1024)
    p.add_argument("--max_model_len_super", type=int, default=8192)
    p.add_argument("--max_model_len_fallback", type=int, default=4096)

    p.add_argument(
        "--model_path",
        required=True,
        help="Generative model: Hugging Face model id or local directory for vLLM",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--run_data_num", type=int, default=-1)
    p.add_argument("--retriever_model_path", default="facebook/contriever")
    p.add_argument("--faiss_index_path", default=_DEFAULT_FAISS)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--think_mode", action="store_true")
    p.add_argument(
        "--lora_meta",
        default=None,
        help="PEFT adapter for Meta-Planner (vLLM LoRA)",
    )
    p.add_argument("--lora_supervisor", default=None, help="PEFT adapter for Supervisor")
    p.add_argument("--lora_extractor", default=None, help="PEFT adapter for Extractor")
    p.add_argument("--lora_reasoner", default=None, help="PEFT adapter for Reasoner")
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument(
        "--max_loras",
        type=int,
        default=8,
        help="vLLM max concurrent LoRA slots",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run_name = args.run_name or default_run_name(args.input_jsonl)
    plan_bn = plan_file_basename(args)
    print(f"[stride] run_name={args.run_name} meta-plan file: {plan_bn}", flush=True)

    if not args.skip_meta:
        run_meta_plan(args)
    else:
        print("[stride] Skipping meta-plan.", flush=True)

    if not args.skip_supervisor:
        run_supervisor(args, plan_bn)
    else:
        print("[stride] Skipping supervisor.", flush=True)

    if args.run_fallback:
        used = used_result_relpath(args, plan_bn)
        print(f"[stride] Fallback uses supervisor output: {used}", flush=True)
        run_fallback(args, used, plan_bn)
    else:
        print("[stride] Fallback skipped. Evaluate: python -m run_eval <jsonl>", flush=True)


if __name__ == "__main__":
    main()
