"""
Build STRIDE-FT training jsonl from pipeline trajectory logs (four modules).

**Supervised (SFT):** Reasoner, Supervisor, Extractor → ``instruction`` / ``input`` /
``output`` → ``ft_preprocess`` → ``lora_ft``.

**Extractor** uses two steps: ``extractor-intermediate`` (mine sub-questions + fact
lists from logs only) then ``extractor-sft`` (attach Contriever top-k documents and
the extractor prompt). Optional LLM **fact minimization** is offline: add
``minimized_context`` to intermediate rows, then ``extractor-sft --context_field
minimized_context``.

**Meta-Planner (DPO):** ``meta-dpo`` reads **K** aligned supervisor outputs and **K**
meta-plan jsonl files (same ids / line order per trajectory), scores trajectories,
and writes ``prompt`` / ``chosen`` / ``rejected`` for ``lora_dpo``.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import jsonlines
from tqdm import tqdm

from contriever_model import load_contriever_and_tokenizer
from metrics import check_none_answer, convert_boolean_answer
from my_retriever import DenseRetriever
from paths import stride_root
from utils import cover_em_score, exact_match_score, f1_score


def _extract_plans(meta_plan: str) -> str:
    """Strip meta text down to the concrete plan block (same logic as ``supervisor``)."""
    try:
        need_str = re.findall(r"(Concrete Plan:.*)", meta_plan, re.DOTALL)[0].strip()
        return need_str.replace("Concrete Plan:", "Plan:")
    except IndexError:
        return meta_plan.strip()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for r in rows:
            writer.write(r)


def _maybe_truncate(rows: List[Dict[str, Any]], max_examples: int) -> List[Dict[str, Any]]:
    if max_examples and max_examples > 0:
        return rows[:max_examples]
    return rows


def parse_progress_str(p_str: str) -> Dict[str, Any]:
    lines = p_str.strip().split("\n")
    result: Dict[str, Any] = {"solved": {}, "pending": [], "failure_log": {}}
    for line in lines:
        line = line.strip()
        if line.startswith("Solved: "):
            s = line[8:].strip()
            if s and s != "{}":
                try:
                    result["solved"] = json.loads(s.replace("'", '"'))
                except json.JSONDecodeError:
                    try:
                        result["solved"] = ast.literal_eval(s)
                    except (ValueError, SyntaxError):
                        result["solved"] = {}
        elif line.startswith("Pending: "):
            s = line[9:].strip()
            if s and s != "[]":
                try:
                    result["pending"] = json.loads(s.replace("'", '"'))
                except json.JSONDecodeError:
                    try:
                        result["pending"] = ast.literal_eval(s)
                    except (ValueError, SyntaxError):
                        result["pending"] = []
        elif line.startswith("FailureLog: "):
            s = line[12:].strip()
            if s and s != "{}":
                try:
                    result["failure_log"] = json.loads(s.replace("'", '"'))
                except json.JSONDecodeError:
                    try:
                        result["failure_log"] = ast.literal_eval(s)
                    except (ValueError, SyntaxError):
                        result["failure_log"] = {}
    return result


def extract_successful_rewrite_turns(
    progress_str_list: List[str], output_str_list: List[str]
) -> List[Tuple[str, str]]:
    n = len(output_str_list)
    if n == 0:
        return []
    parsed_progress = [parse_progress_str(p) for p in progress_str_list]
    all_failure_queries: Dict[str, Set[str]] = {}
    for p in parsed_progress:
        fl = p["failure_log"]
        for qid, queries in fl.items():
            if isinstance(queries, list):
                all_failure_queries.setdefault(qid, set()).update(queries)
            else:
                all_failure_queries.setdefault(qid, set()).add(queries)

    successful_pairs: List[Tuple[str, str]] = []
    for i in range(n):
        try:
            actions = json.loads(output_str_list[i].strip())
        except json.JSONDecodeError:
            continue
        rewrite_actions = [a for a in actions if a.get("action") == "rewrite"]
        if not rewrite_actions:
            continue
        is_last_round = i == n - 1
        for act in rewrite_actions:
            qid = act.get("qid")
            rewritten_query = act.get("query", "")
            if not qid:
                continue
            success = False
            if not is_last_round:
                for j in range(i + 1, n):
                    solved = parsed_progress[j]["solved"]
                    pending = parsed_progress[j]["pending"]
                    if qid in solved and qid not in pending:
                        success = True
                        break
            else:
                if len(actions) == 1:
                    failed_queries = all_failure_queries.get(qid, set())
                    if rewritten_query not in failed_queries:
                        success = True
            if success:
                successful_pairs.append((progress_str_list[i], output_str_list[i]))
                break
    return successful_pairs


def parse_fact_string(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    chunks = re.findall(r"\[\s*\"((?:\\.|[^\"])*)\"\s*\]", raw)
    if chunks:
        out: List[str] = []
        for c in chunks:
            try:
                out.append(json.loads(f'"{c}"').strip())
            except json.JSONDecodeError:
                out.append(c.replace('\\"', '"').strip())
        return [x for x in out if x]
    return [raw] if raw else []


def build_extractor_intermediate_rows(
    records: Iterable[Dict[str, Any]], corpus_name: str
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for record in records:
        final_answer = (
            str(record["final_answer"]) if record.get("final_answer") is not None else ""
        )
        label = record.get("label", "")
        if final_answer == "":
            continue
        if str(label).lower() in ("yes", "no"):
            final_answer = convert_boolean_answer(final_answer)
        em = exact_match_score(final_answer, label)
        c_em = cover_em_score(final_answer, label)
        f1, _, _ = f1_score(final_answer, str(label))
        if c_em != 1.0 and f1 <= 0.5:
            continue
        fact_dict = record.get("fact_dict") or {}
        reasoner_logs = {r[0]: r[-1] for r in record.get("reasoner_records") or []}
        for qid, pair in fact_dict.items():
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            sub_query, raw_facts_str = pair[0], pair[1]
            gold_ans = reasoner_logs.get(qid)
            if not isinstance(gold_ans, list):
                gold_ans = str(gold_ans) if gold_ans is not None else ""
                if not gold_ans:
                    continue
                if check_none_answer(gold_ans):
                    continue
                if len(gold_ans.split()) > 15:
                    continue
            facts_list = parse_fact_string(str(raw_facts_str))
            if len(facts_list) <= 1:
                continue
            samples.append(
                {
                    "question": sub_query,
                    "context": facts_list,
                    "answer": gold_ans,
                    "dataset": corpus_name,
                    "source_id": record.get("id"),
                    "source_qid": qid,
                    "source_em": float(em),
                    "source_c_em": float(c_em),
                    "source_f1": float(f1),
                }
            )
    return samples


def align_reasoner_output(output: str, label: str) -> Optional[str]:
    """Replace the JSON ``answer`` field with ``label`` (escaped for JSON)."""
    esc = json.dumps(str(label))[1:-1]
    replaced, n = re.subn(
        r'"answer"\s*:\s*"(.*?)"',
        f'"answer": "{esc}"',
        output,
        count=1,
        flags=re.DOTALL,
    )
    if n:
        return replaced
    lab_sq = str(label).replace("\\", "\\\\").replace("'", "\\'")
    replaced, n = re.subn(
        r"'answer'\s*:\s*'(.*?)'",
        f"'answer': '{lab_sq}'",
        output,
        count=1,
        flags=re.DOTALL,
    )
    if n:
        return replaced
    return None


def collect_reasoner_ft_rows(
    records: Iterable[Dict[str, Any]],
    *,
    reasoner_prompt: str,
    positive_multiplier: int,
    seed: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    for item in records:
        label = item.get("label", "")
        predict = str(item.get("final_answer") or "").strip()
        if predict == "":
            continue
        if str(label).lower() in ("yes", "no"):
            predict = convert_boolean_answer(predict)
        em = exact_match_score(predict, label)
        c_em = cover_em_score(predict, label)
        f1, _, _ = f1_score(predict, str(label))
        recs = item.get("reasoner_records") or []
        if not recs:
            continue
        inp, out = recs[-1][1], recs[-1][2]
        if isinstance(item.get("final_answer"), list):
            continue
        row = {
            "id": item.get("id"),
            "query": item.get("query"),
            "label": label,
            "final_answer": item.get("final_answer"),
            "input": inp,
            "output": out,
        }
        if em == 1.0:
            positives.append(row)
            continue
        if c_em != 1.0 or em == 1.0:
            continue
        trivial = (
            f"{label}, " in predict or f"and {label}" in predict or f", {label}" in predict
        )
        if trivial or f1 == 0.0:
            continue
        negatives.append(row)
    picked_pos: List[Dict[str, Any]] = []
    if negatives and positives:
        k = min(len(positives), len(negatives) * positive_multiplier)
        if k > 0:
            picked_pos = random.sample(positives, k)
    merged = picked_pos + negatives
    if not merged:
        print(
            "No reasoner rows after filtering (no negatives, or no positives sampled).",
            file=sys.stderr,
        )
    ft: List[Dict[str, Any]] = []
    for example in merged:
        aligned = align_reasoner_output(example["output"], str(example["label"]))
        if aligned is None:
            continue
        ft.append(
            {
                "instruction": "",
                "input": reasoner_prompt.rstrip() + "\n" + example["input"],
                "output": aligned,
            }
        )
    return ft


def convert_extractor_rows_to_ft(
    samples: List[Dict[str, Any]],
    *,
    extractor_prompt: str,
    retriever_model_path: str,
    faiss_index_pattern: str,
    context_field: str,
    top_k_map: Dict[str, int],
    device: str | None = None,
) -> List[Dict[str, Any]]:
    model, tokenizer = load_contriever_and_tokenizer(retriever_model_path)
    if device:
        model = model.to(device)
    retriever = DenseRetriever(model, tokenizer)
    current_dataset: Optional[str] = None
    ft_rows: List[Dict[str, Any]] = []
    for example in tqdm(samples, desc="extractor-sft"):
        ds = example.get("dataset") or "hotpotqa"
        if ds != current_dataset:
            current_dataset = ds
            index_path = faiss_index_pattern.replace("{dataset}", ds)
            retriever.load_index(index_path)
        question = example["question"]
        top_k = top_k_map.get(ds, 5)
        retrieval_result = retriever.retrieve(question, top_k=top_k)
        doc_str = ""
        for res in retrieval_result:
            doc_str += f"Title: {res['title']}\n{res['text']}\n\n"
        e_post = f"Question: \n{question}\n\nDocuments: \n{doc_str}"
        ctx = example.get(context_field)
        if ctx is None:
            raise KeyError(
                f"Missing '{context_field}' in sample; keys={list(example.keys())}"
            )
        e_output = str(ctx)
        ft_rows.append(
            {
                "instruction": extractor_prompt,
                "input": e_post,
                "output": e_output,
            }
        )
    return ft_rows


def convert_supervisor_rows_to_ft(
    cases: List[Dict[str, Any]], supervisor_prompt: str
) -> List[Dict[str, Any]]:
    ft_datas: List[Dict[str, Any]] = []
    for example in cases:
        question = example["question"]
        plan = _extract_plans(example["plan"])
        progress = example["progress"]
        output = example["output"]
        s_post = f"Question: {question}\n\n{plan}\n\nProgress: \n{progress}"
        s_output = f"```json\n{output}\n```"
        ft_datas.append(
            {
                "instruction": supervisor_prompt,
                "input": s_post,
                "output": s_output,
            }
        )
    return ft_datas


def collect_supervisor_cases(
    supervisor_rows: List[Dict[str, Any]],
    plan_by_id: Dict[str, str],
) -> List[Dict[str, Any]]:
    successful: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for item in supervisor_rows:
        final_answer = str(item.get("final_answer") or "")
        if final_answer == "":
            continue
        label = item.get("label", "")
        if str(label).lower() in ("yes", "no"):
            final_answer = convert_boolean_answer(final_answer)
        c_em = cover_em_score(final_answer, label)
        if c_em != 1.0:
            continue
        supervisor_records = item.get("supervisor_records") or {}
        progress = supervisor_records.get("progress") or []
        output = supervisor_records.get("output") or []
        if not any(parse_progress_str(p).get("failure_log") for p in progress):
            continue
        pairs = extract_successful_rewrite_turns(progress, output)
        if not pairs:
            continue
        for p_str, o_str in pairs:
            if not parse_progress_str(p_str).get("failure_log"):
                continue
            eid = item.get("id")
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            pred = plan_by_id.get(str(eid))
            if pred is None:
                continue
            successful.append(
                {
                    "id": eid,
                    "question": item.get("query", ""),
                    "progress": p_str,
                    "output": o_str,
                    "plan": pred,
                    "dataset": item.get("dataset", ""),
                }
            )
            break
    return successful


def _none_fact_ratio(record: Dict[str, Any]) -> float:
    facts = record.get("extracted_facts") or []
    if not facts:
        return 0.0
    none_count = 0
    for fact in facts:
        if isinstance(fact, (list, tuple)) and len(fact) >= 2 and fact[1] == "None":
            none_count += 1
    return round(none_count / len(facts), 2)


def _solved_pending_failure(record: Dict[str, Any]) -> Tuple[int, float, int]:
    recs = record.get("supervisor_records") or {}
    progress = recs.get("progress") or []
    if not progress:
        return 0, 0.0, 0
    last = parse_progress_str(progress[-1])
    solved = last.get("solved") or {}
    pending = last.get("pending") or []
    fl = last.get("failure_log") or {}
    solved_count = len(solved) if isinstance(solved, dict) else 0
    if isinstance(pending, list):
        pending_count = len(pending)
    else:
        pending_count = str(pending).count("Q") if pending else 0
    failure_count = 0
    if isinstance(fl, dict):
        for v in fl.values():
            failure_count += len(v) if isinstance(v, list) else 1
    qc = solved_count + pending_count
    solved_ratio = round(solved_count / qc, 2) if qc > 0 else 0.0
    return solved_count, solved_ratio, failure_count


def _meta_trajectory_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    label = record.get("label", "")
    predict = str(record.get("final_answer") or "").strip()
    if predict == "":
        em, c_em, f1v = 0, 0, 0.0
        answer_flag = 0
    else:
        if str(label).lower() in ("yes", "no"):
            predict = convert_boolean_answer(predict)
        em = int(exact_match_score(predict, label))
        c_em = int(cover_em_score(predict, label))
        f1v, _, _ = f1_score(predict, str(label))
        answer_flag = 1
    f1v = round(float(f1v), 2)
    it = int(record.get("iteration") or 0)
    none_fact = _none_fact_ratio(record)
    _, solved_ratio, failure_count = _solved_pending_failure(record)
    return {
        "em": em,
        "c_em": c_em,
        "f1": f1v,
        "answer_flag": answer_flag,
        "iter": it,
        "none_fact": none_fact,
        "solved_ratio": solved_ratio,
        "failure_count": failure_count,
    }


def is_plan_different(plan_a: str, plan_b: str, threshold: float = 0.85) -> bool:
    f1_val = f1_score(plan_a, plan_b)[0]
    q_count_a = len(re.findall(r"Q\d+:", plan_a))
    q_count_b = len(re.findall(r"Q\d+:", plan_b))
    step_count_a = plan_a.count("Step")
    step_count_b = plan_b.count("Step")
    return (f1_val < threshold) or (q_count_a != q_count_b) or (step_count_a != step_count_b)


def select_meta_dpo_pair(
    metrics_per_traj: List[Dict[str, Any]],
    plans: List[str],
    *,
    num_traj: int,
    plan_diff_threshold: float,
    top_good: int,
    top_bad: int,
) -> Optional[Tuple[int, int]]:
    """Return (chosen_idx, rejected_idx) as 0-based trajectory indices, or None."""
    is_answer_list = [m["answer_flag"] for m in metrics_per_traj]
    em_list = [m["em"] for m in metrics_per_traj]
    c_em_list = [m["c_em"] for m in metrics_per_traj]
    f1_list = [m["f1"] for m in metrics_per_traj]
    iter_list = [m["iter"] for m in metrics_per_traj]
    none_fact_list = [m["none_fact"] for m in metrics_per_traj]
    solved_list = [m["solved_ratio"] for m in metrics_per_traj]
    failure_list = [m["failure_count"] for m in metrics_per_traj]

    has_any_meaningful = any(
        is_answer_list[t] == 1 and (c_em_list[t] == 1 or f1_list[t] >= 0.5)
        for t in range(num_traj)
    )
    if not has_any_meaningful:
        return None
    if sum(em_list) == num_traj or sum(c_em_list) == num_traj:
        return None
    if all(f1_list[t] == f1_list[0] for t in range(num_traj)):
        return None

    if any(em_list):
        good_candidates = [t for t in range(num_traj) if em_list[t] == 1]
    elif any(c_em_list):
        good_candidates = [t for t in range(num_traj) if c_em_list[t] == 1]
    else:
        good_candidates = [
            t for t in range(num_traj) if is_answer_list[t] == 1 and f1_list[t] >= 0.5
        ]
    if not good_candidates:
        return None

    def good_score(t: int) -> Tuple:
        return (
            em_list[t],
            c_em_list[t],
            f1_list[t],
            -iter_list[t],
            -failure_list[t],
            -none_fact_list[t],
        )

    good_candidates.sort(key=good_score, reverse=True)

    bad_candidates = [t for t in range(num_traj) if em_list[t] == 0]
    if not bad_candidates:
        return None

    def bad_score(t: int) -> Tuple:
        is_empty = is_answer_list[t] == 0
        c_em_bad = c_em_list[t] == 0
        return (
            -int(is_empty),
            -int(c_em_bad),
            f1_list[t],
            failure_list[t],
            none_fact_list[t],
            iter_list[t],
        )

    bad_candidates.sort(key=bad_score, reverse=True)

    for g in good_candidates[:top_good]:
        for b in bad_candidates[:top_bad]:
            if g == b:
                continue
            if not is_plan_different(plans[g], plans[b], plan_diff_threshold):
                continue
            if em_list[g] != em_list[b]:
                return g, b
            if f1_list[g] - f1_list[b] >= 0.2:
                return g, b
            if c_em_list[g] != c_em_list[b]:
                return g, b
    return None


def build_meta_dpo_rows(
    supervisor_paths: Sequence[Path],
    meta_plan_paths: Sequence[Path],
    *,
    meta_system_prompt: str,
    max_examples: int,
    plan_diff_threshold: float,
    top_good: int,
    top_bad: int,
) -> List[Dict[str, str]]:
    t = len(supervisor_paths)
    if t != len(meta_plan_paths):
        raise ValueError(
            f"Need equal trajectory counts; got {t} supervisor vs {len(meta_plan_paths)} meta_plan"
        )
    if t < 2:
        raise ValueError("Meta-DPO needs at least two trajectories.")

    sup_lists = [_load_jsonl(p) for p in supervisor_paths]
    meta_lists = [_load_jsonl(p) for p in meta_plan_paths]
    if max_examples is not None and max_examples > 0:
        sup_lists = [rows[:max_examples] for rows in sup_lists]
        meta_lists = [rows[:max_examples] for rows in meta_lists]
    n = min(len(x) for x in sup_lists)
    n = min(n, min(len(x) for x in meta_lists))
    out: List[Dict[str, str]] = []
    id_mismatch_warned = False
    for i in range(n):
        rows = [sup_lists[k][i] for k in range(t)]
        metas = [meta_lists[k][i] for k in range(t)]
        id0 = rows[0].get("id")
        if any(r.get("id") != id0 for r in rows) or any(m.get("id") != id0 for m in metas):
            if not id_mismatch_warned:
                print(
                    "Warning: id mismatch at some row index; skipping those rows. "
                    "Regenerate jsonl with identical line order per trajectory.",
                    file=sys.stderr,
                )
                id_mismatch_warned = True
            continue
        metrics_per_traj = [_meta_trajectory_metrics(r) for r in rows]
        plans = [str(m.get("predict") or "") for m in metas]
        pair = select_meta_dpo_pair(
            metrics_per_traj,
            plans,
            num_traj=t,
            plan_diff_threshold=plan_diff_threshold,
            top_good=top_good,
            top_bad=top_bad,
        )
        if pair is None:
            continue
        g, b = pair
        question = str(rows[0].get("query") or "")
        chosen_output = plans[g]
        rejected_output = plans[b]
        user_input = f"Question: \n{question}"
        prompt = (
            f"<|im_start|>system\n{meta_system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        out.append(
            {"prompt": prompt, "chosen": chosen_output, "rejected": rejected_output}
        )
    return out


def cmd_meta_dpo(ns: argparse.Namespace) -> None:
    if len(ns.supervisor_traj) != len(ns.meta_plan_traj):
        print(
            "Error: pass the same number of --supervisor_traj and --meta_plan_traj "
            f"(got {len(ns.supervisor_traj)} vs {len(ns.meta_plan_traj)}).",
            file=sys.stderr,
        )
        sys.exit(1)
    system = Path(ns.meta_system_prompt).read_text(encoding="utf-8")
    rows = build_meta_dpo_rows(
        ns.supervisor_traj,
        ns.meta_plan_traj,
        meta_system_prompt=system,
        max_examples=ns.max_examples,
        plan_diff_threshold=ns.plan_diff_threshold,
        top_good=ns.top_good,
        top_bad=ns.top_bad,
    )
    _write_jsonl(Path(ns.output_jsonl), rows)
    print(f"Wrote {len(rows)} Meta-Planner DPO rows to {ns.output_jsonl}")


def cmd_reasoner(ns: argparse.Namespace) -> None:
    prompt = Path(ns.reasoner_prompt).read_text(encoding="utf-8")
    all_rows: List[Dict[str, Any]] = []
    for p in ns.input_jsonl:
        all_rows.extend(_load_jsonl(Path(p)))
    all_rows = _maybe_truncate(all_rows, ns.max_examples)
    ft = collect_reasoner_ft_rows(
        all_rows,
        reasoner_prompt=prompt,
        positive_multiplier=ns.positive_multiplier,
        seed=ns.seed,
    )
    _write_jsonl(Path(ns.output_jsonl), ft)
    print(f"Wrote {len(ft)} reasoner SFT rows to {ns.output_jsonl}")


def cmd_extractor_intermediate(ns: argparse.Namespace) -> None:
    all_rows: List[Dict[str, Any]] = []
    for p in ns.input_jsonl:
        all_rows.extend(_load_jsonl(Path(p)))
    all_rows = _maybe_truncate(all_rows, ns.max_examples)
    samples = build_extractor_intermediate_rows(all_rows, ns.corpus_name)
    _write_jsonl(Path(ns.output_jsonl), samples)
    print(f"Wrote {len(samples)} extractor intermediate rows to {ns.output_jsonl}")


def cmd_extractor_sft(ns: argparse.Namespace) -> None:
    samples = _maybe_truncate(_load_jsonl(Path(ns.input_jsonl)), ns.max_examples)
    extractor_prompt = Path(ns.extractor_prompt).read_text(encoding="utf-8")
    pattern = ns.faiss_index_pattern
    if "{dataset}" not in pattern:
        print(
            "Warning: --faiss_index_pattern has no {dataset} placeholder; "
            "the same directory is reused for every sample.",
            file=sys.stderr,
        )
    top_k_map = {
        "2wikimultihopqa": ns.top_k_2wiki,
        "hotpotqa": ns.top_k_hotpot,
        "musique": ns.top_k_musique,
    }
    ft = convert_extractor_rows_to_ft(
        samples,
        extractor_prompt=extractor_prompt,
        retriever_model_path=ns.retriever_model_path,
        faiss_index_pattern=pattern,
        context_field=ns.context_field,
        top_k_map=top_k_map,
        device=ns.device,
    )
    _write_jsonl(Path(ns.output_jsonl), ft)
    print(f"Wrote {len(ft)} extractor SFT rows to {ns.output_jsonl}")


def cmd_supervisor(ns: argparse.Namespace) -> None:
    prompt = Path(ns.supervisor_prompt).read_text(encoding="utf-8")
    all_cases: List[Dict[str, Any]] = []
    for sup_path, plan_path in ns.run:
        sup_rows = _maybe_truncate(_load_jsonl(Path(sup_path)), ns.max_examples)
        plan_rows = _load_jsonl(Path(plan_path))
        plan_by_id = {str(x["id"]): x.get("predict", "") for x in plan_rows}
        all_cases.extend(collect_supervisor_cases(sup_rows, plan_by_id))
    ft = convert_supervisor_rows_to_ft(all_cases, prompt)
    _write_jsonl(Path(ns.output_jsonl), ft)
    print(f"Wrote {len(ft)} supervisor SFT rows to {ns.output_jsonl}")


def build_arg_parser() -> argparse.ArgumentParser:
    root = stride_root()
    p = argparse.ArgumentParser(description="Build STRIDE-FT jsonl from trajectory logs")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("reasoner", help="Reasoner SFT jsonl from supervisor output jsonl")
    pr.add_argument("--input_jsonl", action="append", required=True)
    pr.add_argument("--output_jsonl", type=Path, required=True)
    pr.add_argument(
        "--reasoner_prompt",
        type=Path,
        default=root / "prompt" / "reasoner" / "default.txt",
    )
    pr.add_argument("--positive_multiplier", type=int, default=2)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Use only the first N merged input rows (paper: 5000 on a 10k pool). 0 = no limit.",
    )
    pr.set_defaults(func=cmd_reasoner)

    pe = sub.add_parser(
        "extractor-intermediate",
        help="Stage 1/2 for Extractor only: mine (question, fact list, answer) from logs (no retrieval)",
    )
    pe.add_argument("--input_jsonl", action="append", required=True)
    pe.add_argument("--corpus_name", required=True, help="Tag stored as dataset / FAISS key")
    pe.add_argument("--output_jsonl", type=Path, required=True)
    pe.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Use only the first N merged input rows (paper: 5000). 0 = no limit.",
    )
    pe.set_defaults(func=cmd_extractor_intermediate)

    pes = sub.add_parser(
        "extractor-sft",
        help="Stage 2/2 for Extractor: SFT jsonl from intermediate rows + Contriever + FAISS",
    )
    pes.add_argument("--input_jsonl", type=Path, required=True)
    pes.add_argument("--output_jsonl", type=Path, required=True)
    pes.add_argument(
        "--extractor_prompt",
        type=Path,
        default=root / "prompt" / "extractor" / "default.txt",
    )
    pes.add_argument(
        "--faiss_index_pattern",
        type=str,
        default=str(root / "faiss_index" / "{dataset}" / "index"),
        help="Path with literal {dataset} replaced per sample's dataset field",
    )
    pes.add_argument("--retriever_model_path", default="facebook/contriever")
    pes.add_argument(
        "--context_field",
        default="context",
        choices=("context", "minimized_context"),
        help="Which field to teach as extractor output",
    )
    pes.add_argument("--top_k_2wiki", type=int, default=3)
    pes.add_argument("--top_k_hotpot", type=int, default=5)
    pes.add_argument("--top_k_musique", type=int, default=3)
    pes.add_argument("--device", default=None, help="Torch device for Contriever (optional)")
    pes.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Use only the first N intermediate rows (paper: 5000). 0 = no limit.",
    )
    pes.set_defaults(func=cmd_extractor_sft)

    ps = sub.add_parser(
        "supervisor",
        help="Supervisor SFT jsonl from supervisor output + meta-plan jsonl pairs",
    )
    ps.add_argument(
        "--run",
        nargs=2,
        metavar=("SUPERVISOR_JSONL", "META_PLAN_JSONL"),
        action="append",
        required=True,
        help="Repeat for each shard (same ids in plan as in supervisor file)",
    )
    ps.add_argument("--output_jsonl", type=Path, required=True)
    ps.add_argument(
        "--supervisor_prompt",
        type=Path,
        default=root / "prompt" / "supervisor" / "default.txt",
    )
    ps.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Use only the first N supervisor rows per --run pair (paper: 5000). 0 = no limit.",
    )
    ps.set_defaults(func=cmd_supervisor)

    pmd = sub.add_parser(
        "meta-dpo",
        help="Meta-Planner DPO jsonl from K parallel supervisor + meta-plan trajectories",
    )
    pmd.add_argument(
        "--supervisor_traj",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="Supervisor output jsonl for trajectory 1, then repeat for 2..K (same line order / ids)",
    )
    pmd.add_argument(
        "--meta_plan_traj",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help="Meta-plan jsonl aligned to each --supervisor_traj (same count and order)",
    )
    pmd.add_argument(
        "--meta_system_prompt",
        type=Path,
        default=root / "prompt" / "meta_plan" / "meta_plan.txt",
    )
    pmd.add_argument("--output_jsonl", type=Path, required=True)
    pmd.add_argument(
        "--max_examples",
        type=int,
        default=5000,
        help="First N aligned lines per trajectory (paper: 5000 on a 10k subsample). Use 0 for no limit.",
    )
    pmd.add_argument("--plan_diff_threshold", type=float, default=0.85)
    pmd.add_argument("--top_good", type=int, default=3)
    pmd.add_argument("--top_bad", type=int, default=5)
    pmd.set_defaults(func=cmd_meta_dpo)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    p = build_arg_parser()
    ns = p.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
