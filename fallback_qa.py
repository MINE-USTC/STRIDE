from vllm import LLM
import numpy as np
import jsonlines
from transformers import AutoTokenizer
import os
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
import re

from stride.contriever_model import load_contriever_and_tokenizer
from stride.my_retriever import DenseRetriever
from stride.paths import default_run_name
from stride.utils import chat_vllm

STRIDE_ROOT = Path(__file__).resolve().parent

def check_none_answer(answer):
    answer = answer.strip().lower() if isinstance(answer, str) else answer
    if answer is None:
        return True
    if isinstance(answer, str) and answer in ['none', 'no ', 'n/a', 'not mentioned', 'not given', 'unknown', '']:
        return True
    if isinstance(answer, list) and all(isinstance(a, str) and a.strip().lower() in ['none', 'no', 'n/a', 'not mentioned', 'not given', 'unknown', ''] for a in answer):
        return True
    if isinstance(answer, str) and re.search(r'\b(none|no |n/a|not mentioned|not given|unknown)\b', answer):
        return True
    return False

### 对检索结果进行排序
def rank_docs(scores, texts, titles):
    sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
    ranked_texts = [texts[i] for i in sorted_indices]
    ranked_titles = [titles[i] for i in sorted_indices]
    ranked_scores = [scores[i] for i in sorted_indices]
    return ranked_texts, ranked_titles, ranked_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default='0', type=str, help="gpu id"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Generative model: Hugging Face model id or local directory for vLLM",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subfolder under meta_plans/ and output/ (default: stem of input_jsonl); must match main run",
    )
    parser.add_argument(
        "--index_corpus",
        type=str,
        default=None,
        help="Replaces 'dataset' in --faiss_index_path (default: same as run_name)",
    )
    parser.add_argument(
        "--write_file_name", default='fallback_qa', type=str, help="output file base name"
    )
    parser.add_argument(
        '--top_k_docs', type=int, default=5, help="top k retrieved documents"
    )
    parser.add_argument(
        '--retriever_model_path',
        type=str,
        default='facebook/contriever',
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        '--faiss_index_path', type=str, default=None, help="default: stride/faiss_index/dataset/index"
    )
    parser.add_argument(
        '--use_qwen3', type=bool, default=False, help="Whether to use qwen3"
    )
    parser.add_argument(
        '--think_mode', type=bool, default=False, help="Whether to use think mode for qwen3"
    )
    parser.add_argument(
        '--plan_file_name', type=str, default='meta_plan.jsonl', help='Under meta_plans/<run_name>/',
    )
    parser.add_argument(
        '--used_result_file', type=str, default='plan/stride_top5.jsonl', help='Relative to stride/output/<run_name>/ (first segment = meta_plan version dir)',
    )
    ### VLLM 参数
    parser.add_argument(
        '--max_model_len', default=8192, type=int, help='max_model_len (prompt+output) for Vllm init'
    )
    parser.add_argument(
        '--max_num_seqs', default=64, type=int, help='max_num_seqs for Vllm init'
    )
    parser.add_argument(
        '--gpu_memory_utilization', default=0.85, type=float, help='gpu_memory_utilization for Vllm init'
    )
    parser.add_argument(
        '--tensor_parallel_size', default=1, type=int, help='tensor_parallel_size for Vllm init'
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Optional: restrict to example ids listed in this jsonl",
    )
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = default_run_name(args.input_jsonl) if args.input_jsonl else None
    if not args.run_name:
        raise SystemExit(
            "fallback_qa: set --run_name to the same value as the main pipeline run, "
            "or pass --input_jsonl so the run name can be inferred from its stem."
        )
    index_corpus = args.index_corpus if args.index_corpus else args.run_name
    if args.faiss_index_path is None:
        args.faiss_index_path = str(STRIDE_ROOT / "faiss_index" / "dataset" / "index")

    '''
    Fallback Reasoner (paper): answer when the main run has no final answer — plan + facts + retrieval.
    '''
    
    batch_size = args.batch_size
    model_path = args.model_path

    model = LLM(
        model=model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    
    if 'Qwen3' in model_path:
        args.use_qwen3 = True
    else:
        args.think_mode = None
        
    
    ### 加载检索模型和faiss index
    used_contriever, used_contriever_tokenizer = load_contriever_and_tokenizer(
        args.retriever_model_path
    )
    
    used_retriever = DenseRetriever(used_contriever, used_contriever_tokenizer)
    
    faiss_index_path = args.faiss_index_path.replace('dataset', index_corpus)
    used_retriever.load_index(faiss_index_path)
    
    ### prompt文件
    prompt_path = STRIDE_ROOT / "prompt" / "fallback_qa.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        full_prompt = f.read()
    ### prompt留的位置，用于后续替换
    document_str = r"Documents content here"
    question_str = r"Question here"
    plan_str = r"Plan content here"
    fact_str = r"Facts here"
        
    ### 写入文件的文件名
    write_file_name = args.write_file_name

    write_file_name = args.used_result_file.replace('.jsonl', '') + '-' + write_file_name
    write_file_prefolder = write_file_name.split('/')[0]
    write_file_name = write_file_name.split('/')[-1]
    
    print(f"Write file name: {write_file_name}, in folder: {write_file_prefolder}")
    
    run_name = args.run_name
    
    note = f'fallback_qa used_result={args.used_result_file}, faiss_index_path={faiss_index_path}, docs={used_retriever.ctr}, top_k={args.top_k_docs}'
    note += f', model_path={model_path}, batch_size={batch_size}'
    note += f', prompt文件: {prompt_path}, use_qwen3={args.use_qwen3}, think_mode={args.think_mode}'
    note += f', plan_file={args.plan_file_name}'
    
    
    plan_path = str(STRIDE_ROOT / "meta_plans" / run_name / args.plan_file_name)
    plan_dict = {}
    with jsonlines.open(plan_path, 'r') as reader:
        for item in reader:
            plan_dict[item['id']] = item['predict']
    print(f"Load plans from {plan_path}, total {len(plan_dict)} plans!")
    
    print(note)
    
    total_data = []
    used_result_path = str(STRIDE_ROOT / "output" / run_name / args.used_result_file)
    allow_ids = None
    if args.input_jsonl:
        allow_ids = set()
        with jsonlines.open(str(Path(args.input_jsonl).expanduser().resolve()), "r") as reader:
            for item in reader:
                allow_ids.add(item["id"])
    with jsonlines.open(used_result_path, 'r') as reader:
        for item in reader:
            if item['final_answer'] is None or check_none_answer(item['final_answer']):
                if allow_ids is not None and item["id"] not in allow_ids:
                    continue
                total_data.append(item)
    print(f"Load data from {used_result_path}, total {len(total_data)} data!")

    ### 写入文件的path
    write_path = str(
        STRIDE_ROOT / "output" / run_name / write_file_prefolder / f"{write_file_name}.jsonl"
    )
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path))
    
    exist_data = []
    if os.path.exists(write_path):
        with jsonlines.open(write_path) as reader:
            for obj in reader:
                exist_data.append(obj['id'])
    print(f"Already exist {len(exist_data)} sentences in {write_path}!")
    
    output_folder = str(STRIDE_ROOT / "output" / run_name / write_file_prefolder / "log")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 新建txt文件并指定路径
    output_file = os.path.join(output_folder, f'{write_file_name}.txt')
    
    # 打开文件准备写入内容
    print("Ready to process main task!")
    with open(output_file, 'a') as f:
        # 重定向标准输出到txt文件
        sys.stdout = f
        if len(exist_data) == 0:
            print("PROMPT:\n" + full_prompt)
            print('-------------------------------------------\n')
        with jsonlines.open(write_path, 'a') as writer:
            for start_idx in tqdm(range(0, len(total_data), batch_size), desc=f"Fallback QA [{run_name}]"):
                end_idx = min(start_idx + batch_size, len(total_data))
                batch_data = total_data[start_idx:end_idx]
                
                messages_list = []
                input_list = []
                item_list = []
                for item in batch_data:
                    id_ = item['id']
                    if id_ in exist_data:
                        continue
                    item_list.append(item)
                    question = item['query']
                    
                    messages = []
                    
                    doc_str = ""
                    
                    fact_dict = item['fact_dict']
                    facts = "" 
                    last_key = None
                    if len(fact_dict) > 0:
                        facts = "Facts: \n"
                        for k, v in fact_dict.items():
                            facts += f"- {k}: {v}\n"
                            last_key = k
                        facts = facts.strip()
                        titles, texts, scores = [], [], [] ### 该子问题的facts检索到的所有内容
                        for fact in fact_dict[last_key]:
                            retrieve_info = used_retriever.retrieve(fact, top_k=args.top_k_docs)
                            for res in retrieve_info:
                                if res['title'] in titles:
                                    ### 找到下标，title一样，但是text可能不一样
                                    idx = titles.index(res['title'])
                                    if res['text'][:20] != texts[idx][:20]:
                                        ### text也不一样
                                        titles.append(res['title'])
                                        texts.append(res['text'])
                                        scores.append(res['score'])
                                    else:
                                        continue
                                else: ### title不一样
                                    titles.append(res['title'])
                                    texts.append(res['text'])
                                    scores.append(res['score'])
                        rank_texts, rank_titles, rank_scores = rank_docs(scores, texts, titles)
                        for title, text in zip(rank_titles[:args.top_k_docs], rank_texts[:args.top_k_docs]):
                            doc_str += f"Title: {title}\n{text}\n\n"
                        doc_str = doc_str.strip()
                    else: ### 没有fact，就用原始question来检索
                        retrieval_result = used_retriever.retrieve(question, top_k=args.top_k_docs)
                        for res in retrieval_result:
                            title = res['title']
                            text = res['text']
                            score = res['score']
                            doc_str += f"Title: {title}\n{text}\n\n"
                        doc_str = doc_str.strip()
                           
                    qa_prompt = full_prompt.replace(document_str, doc_str).replace(question_str, question).replace(fact_str, facts)
                    
                    if id_ in plan_dict:
                        plan_content = plan_dict[id_]
                    else:
                        plan_content = ""
                        print(f"Warning: {id_} not in plan dict!")
                    qa_prompt = qa_prompt.replace(plan_str, plan_content)

                    messages.append({'role': "user", "content": qa_prompt})
                    
                    messages_list.append(messages)
                    input_list.append(qa_prompt)
                    
                if len(messages_list) == 0:
                    continue
                
                generated_texts, input_lengths, output_lengths, times = chat_vllm(
                    messages_list,
                    model,
                    tokenizer,
                    qwen3_think_mode=args.think_mode,
                )
                
                for i, (item, input, generated_text, input_length, output_length) in enumerate(zip(
                    item_list,
                    input_list,
                    generated_texts,
                    input_lengths,
                    output_lengths,
                )):
                    id_ = item['id']
                    question = item['query']
                    time = times/len(generated_texts)
                    
                    print('-------------------------------------------')
                    print(f"ID: {id_}\nInput: \n{input}\n\nAnswer: \n{item['label']}\n\nOutput: \n{generated_text}\n")
                    print('-------------------------------------------')
                    print(f'Input tokens: {input_length}, Output tokens: {output_length}')
                    print(f'Time: {times} seconds')
                    
                    writer.write({
                        "id": id_,
                        'query': question,
                        "label": item['label'],
                        "final_answer": generated_text,
                        "tokens": [input_length, output_length],
                        "time": time,
                        "note": note,
                    })
        ### 恢复标准输出
        sys.stdout.close()
        sys.stdout = sys.__stdout__
