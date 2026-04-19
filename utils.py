import re
import ast
import time
from collections import Counter
import string

from vllm import SamplingParams

def get_dict(text):
    dict = re.findall(r'({.*})', text.replace('\n', ''))[0]
    return eval(dict)['answer']


def get_answer(obj):
    if not obj['predict']:
        return '', True
    flag_error = False
    obj['predict'] = obj['predict'].replace('\n', '')
    try:
        predict = str(get_dict(obj['predict']))
        if '{' in predict:
            temp_dict = ast.literal_eval(predict)
            for key, value in temp_dict.items():
                value = str(value)
                if obj['label'] in value:
                    predict = value
                    break
    except Exception:
        try:
            if 'answer":' in obj['predict']:
                predict = re.findall(r'answer": ?(.*?)}', obj['predict'])[0]
            elif "answer':" in obj['predict']:
                predict = re.findall(r"answer': ?(.*?)}", obj['predict'])[0]
            else:
                predict = ''
        except Exception:
            predict = ''
            flag_error = True
    if obj['label'] in ['yes', 'no']:
        predict = predict.replace('true', 'yes')
        predict = predict.replace('false', 'no')
        predict = re.sub(r'false|False', 'no', predict)
        predict = re.sub(r'true|True', 'yes', predict)
        if 'yes' in predict or 'Yes,' in predict:
            predict = 'yes'
        elif 'no' in predict or 'No,' in predict:
            predict = 'no'
    return str(predict), flag_error


def remove_articles(text):
    text = text.replace('_', ' ')
    text = text.replace('"', '')
    text = re.sub(r"'|`|,|’|\\|´", '', text)
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def white_space_fix(text):
    return ' '.join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower().strip()


def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth) -> float:
    """1.0 if normalized strings match, else 0.0 (same convention as legacy ``utils``)."""
    if prediction is None or ground_truth is None:
        return 0.0
    return (
        1.0
        if normalize_answer(str(prediction)) == normalize_answer(str(ground_truth))
        else 0.0
    )


def cover_em_score(prediction, ground_truth) -> float:
    """
    Cover-style match: every normalized ground-truth token must appear in the
    prediction with at least the same multiplicity (bag-of-tokens on
    ``normalize_answer``). Empty ground truth matches only an empty prediction.
    """
    if ground_truth is None:
        return 0.0
    pred_c = Counter(normalize_answer(str(prediction)).split())
    gt_c = Counter(normalize_answer(str(ground_truth)).split())
    if sum(gt_c.values()) == 0:
        return 1.0 if sum(pred_c.values()) == 0 else 0.0
    for tok, need in gt_c.items():
        if pred_c[tok] < need:
            return 0.0
    return 1.0


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def total_f1_score(predictions, ground_truths):
    total_f1, total_precision, total_recall = 0.0, 0.0, 0.0
    n = len(predictions)

    for i in range(n):
        f1, precision, recall = f1_score(predictions[i], ground_truths[i])
        total_f1 += f1
        total_precision += precision
        total_recall += recall

    num_valid = n
    avg_f1 = total_f1 / num_valid if num_valid > 0 else 0.0
    avg_precision = total_precision / num_valid if num_valid > 0 else 0.0
    avg_recall = total_recall / num_valid if num_valid > 0 else 0.0

    return avg_f1, avg_precision, avg_recall


def total_exact_match_score(y_true, y_pred):
    sum_ = 0
    for true, pred in zip(y_true, y_pred):
        true = normalize_answer(true)
        pred = normalize_answer(pred)
        if true == pred:
            sum_ += 1
    return sum_ / len(y_true)


def chat_vllm(
    messages_list,
    model,
    tokenizer,
    qwen3_think_mode=None,
    params=None,
    lora_request=None,
):
    """
    Batch generation with vLLM. Optionally apply one PEFT adapter via ``lora_request``
    (``vllm.lora.request.LoRARequest``). Pass ``None`` for the base model weights.
    """
    if not params:
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
        )
    else:
        sampling_params = params

    if qwen3_think_mode is not None:
        texts = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=qwen3_think_mode,
        )
    else:
        texts = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
        )

    gen_kw = {"use_tqdm": False}
    if lora_request is not None:
        gen_kw["lora_request"] = lora_request

    t0 = time.perf_counter()
    outputs = model.generate(texts, sampling_params, **gen_kw)
    t1 = time.perf_counter()

    total_prompt_tokens = []
    total_output_tokens = []
    generated_texts = []

    for output in outputs:
        prompt_tokens = len(output.prompt_token_ids)
        generated_tokens = len(output.outputs[0].token_ids)
        total_prompt_tokens.append(prompt_tokens)
        total_output_tokens.append(generated_tokens)
        generated_texts.append(output.outputs[0].text)

    batch_latency = t1 - t0
    return generated_texts, total_prompt_tokens, total_output_tokens, batch_latency
