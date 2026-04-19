"""Evaluation metrics (EM, F1) for STRIDE jsonl outputs."""

from __future__ import annotations

import jsonlines
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import get_answer, total_exact_match_score, total_f1_score


def check_none_answer(answer: Any) -> bool:
    if answer is None:
        return True
    answer = answer.strip().lower() if isinstance(answer, str) else answer
    if isinstance(answer, str) and answer in [
        "none",
        "no ",
        "n/a",
        "not mentioned",
        "not given",
        "unknown",
        "",
    ]:
        return True
    if isinstance(answer, list) and all(
        isinstance(a, str)
        and a.strip().lower()
        in ["none", "no", "n/a", "not mentioned", "not given", "unknown", ""]
        for a in answer
    ):
        return True
    if (
        isinstance(answer, str)
        and re.search(r"\b(none|no |n/a|not mentioned|not given|unknown)\b", answer)
    ):
        return True
    return False


def convert_boolean_answer(predict: str) -> str:
    predict = predict.replace("true", "yes")
    predict = predict.replace("false", "no")
    predict = re.sub(r"false|False", "no", predict)
    predict = re.sub(r"true|True", "yes", predict)
    if "yes" in predict or "Yes," in predict:
        return "yes"
    if "no" in predict or "No," in predict:
        return "no"
    return predict


def prediction_from_record(
    obj: Dict[str, Any],
    *,
    prefer_json_predict: bool = True,
) -> Tuple[str, bool]:
    """
    Extract a single string prediction for scoring.
    Mirrors analysis.ipynb: prefer `final_answer`; fall back to parsed CoT `predict` when empty.
    """
    flag_error = False
    temp_from_predict = ""
    if prefer_json_predict and obj.get("predict"):
        try:
            p, fe = get_answer(dict(obj))
            temp_from_predict = p
            flag_error = fe
        except Exception:
            temp_from_predict = ""

    fa = obj.get("final_answer")
    predict = fa if fa is not None else ""
    predict = str(predict).strip()

    if obj.get("label") in ("yes", "no"):
        predict = convert_boolean_answer(predict)

    if check_none_answer(predict) and temp_from_predict:
        predict = temp_from_predict

    return predict, flag_error


def load_jsonl_path(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            rows.append(obj)
    return rows


def build_predictions_with_optional_fallback(
    main_rows: List[Dict[str, Any]],
    fallback_rows: Optional[List[Dict[str, Any]]],
) -> Tuple[List[str], List[str], int]:
    """
    If fallback_rows is provided, merge by id: use fallback final_answer when main is empty.
    """
    fb_by_id: Dict[str, str] = {}
    if fallback_rows:
        for obj in fallback_rows:
            fa = obj.get("final_answer")
            if fa is None or not str(fa).strip():
                continue
            p = str(fa).strip()
            if obj.get("label") in ("yes", "no"):
                p = convert_boolean_answer(p)
            fb_by_id[obj["id"]] = p

    y_true: List[str] = []
    y_pred: List[str] = []
    err = 0
    for obj in main_rows:
        label = obj.get("label", "")
        pred, fe = prediction_from_record(obj)
        if fe:
            err += 1
        if fb_by_id and check_none_answer(pred) and obj["id"] in fb_by_id:
            pred = fb_by_id[obj["id"]]
        y_true.append(str(label))
        y_pred.append(pred)
    return y_true, y_pred, err


def evaluate_lists(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    em = total_exact_match_score(y_true, y_pred)
    f1, p, r = total_f1_score(y_pred, y_true)
    return {
        "em": float(em),
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "n": float(len(y_true)),
    }


def evaluate_file(
    jsonl_path: Path,
    fallback_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    main_rows = load_jsonl_path(jsonl_path)
    fb_rows = load_jsonl_path(fallback_jsonl) if fallback_jsonl else None
    y_true, y_pred, count_error = build_predictions_with_optional_fallback(main_rows, fb_rows)
    if not y_true:
        return {"error": "empty input", "path": str(jsonl_path)}
    metrics = evaluate_lists(y_true, y_pred)
    metrics["parse_errors"] = float(count_error)
    metrics["path"] = str(jsonl_path)
    if fallback_jsonl:
        metrics["fallback_path"] = str(fallback_jsonl)
    return metrics
