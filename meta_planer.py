import numpy as np
import jsonlines
from transformers import AutoTokenizer
import os
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
from vllm import LLM

from paths import default_run_name
from utils import chat_vllm
from vllm_lora import any_lora_paths, llm_lora_init_kwargs, make_lora_request

STRIDE_ROOT = Path(__file__).resolve().parent

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
        "--input_jsonl",
        required=True,
        type=str,
        help="Path to input jsonl (each row needs id, question)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subfolder under meta_plans/ for this run (default: stem of input_jsonl)",
    )
    parser.add_argument(
        "--write_file_name", default='meta_plan', type=str, help="Output jsonl base name (no suffix for system prompt)"
    )
    parser.add_argument(
        '--prompt_file', type=str, default='meta_plan', help="stem of prompt/meta_plan/<stem>.txt"
    )
    parser.add_argument(
        '--run_data_num', type=int, default=-1, help="Run data num"
    )
    parser.add_argument(
        '--use_qwen3', type=bool, default=False, help="Whether to use qwen3"
    )
    parser.add_argument(
        '--think_mode', type=bool, default=False, help="Whether to use think mode for qwen3"
    )
    parser.add_argument(
        "--lora_meta",
        type=str,
        default=None,
        help="PEFT adapter directory for Meta-Planner (vLLM LoRA)",
    )
    parser.add_argument(
        "--max_lora_rank",
        type=int,
        default=64,
        help="Max LoRA rank (vLLM)",
    )
    parser.add_argument(
        "--max_loras",
        type=int,
        default=8,
        help="Max concurrent LoRA slots in vLLM (meta stage uses one)",
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
    args = parser.parse_args()
    run_name = args.run_name if args.run_name else default_run_name(args.input_jsonl)

    '''
    Meta-Plan
    
    对问题进行plan，让 LLM 自己从问题里抽象出 general strategy，再细化为 concrete plan 和 retrieval queries。
    '''

    batch_size = args.batch_size
    model_path = args.model_path
    use_lora = any_lora_paths(args.lora_meta)

    model = LLM(
        model=model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        **llm_lora_init_kwargs(
            use_lora=use_lora,
            max_lora_rank=args.max_lora_rank,
            max_loras=args.max_loras,
        ),
    )
    req_meta = make_lora_request("meta", 1, args.lora_meta)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    ### prompt文件
    prompt_path = STRIDE_ROOT / "prompt" / "meta_plan" / f"{args.prompt_file}.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        full_prompt = f.read()
    ### prompt留的位置，用于后续替换
    if 'Qwen3' in model_path:
        args.use_qwen3 = True
    else:
        args.think_mode = None
        
    ### 写入文件的文件名
    write_file_name = args.write_file_name

    if use_lora and 'lora' not in write_file_name:
        write_file_name += '-lora'
    
    note = f'model={args.model_path}, batch_size={batch_size}, prompt={prompt_path}'
    
    print(f"Note: {note}")
    
    if args.use_qwen3:
        note += f', think mode = {args.think_mode}'

    path = str(Path(args.input_jsonl).expanduser().resolve())
    with jsonlines.open(path) as reader:
        total_data = [item for item in reader]
    total_data = total_data[:args.run_data_num] if args.run_data_num > 0 else total_data
    print(f"Totally {len(total_data)} questions in {path}!")
    

    ### 写入文件的path
    write_path = str(STRIDE_ROOT / "meta_plans" / run_name / f"{write_file_name}.jsonl")
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path))
    
    exist_data = []
    if os.path.exists(write_path):
        with jsonlines.open(write_path) as reader:
            for obj in reader:
                exist_data.append(obj['id'])
    print(f"Already exist {len(exist_data)} sentences in {write_path}!")
    
    output_folder = str(STRIDE_ROOT / "meta_plans" / run_name / "log")
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
            for start_idx in tqdm(range(0, len(total_data), batch_size), desc=f"Meta-Plan [{run_name}]"):
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
                    question = item['question']
                    
                    messages = []
                    
                    Question_str = f"Question: \n{question}"
                    messages.append({"role": "system", "content": full_prompt})
                    messages.append({"role": "user", "content": Question_str})
                    
                    messages_list.append(messages)
                    input_list.append(Question_str)
                    
                if len(messages_list) == 0:
                    continue

                generated_texts, input_lengths, output_lengths, times = chat_vllm(
                    messages_list,
                    model,
                    tokenizer,
                    qwen3_think_mode=args.think_mode,
                    lora_request=req_meta,
                )

                data_num = len(generated_texts)
                
                for i, (item, input, generated_text, input_length, output_length) in enumerate(zip(
                    item_list,
                    input_list,
                    generated_texts,
                    input_lengths,
                    output_lengths
                )):
                    id_ = item['id']
                    question = item['question']
                    
                    print('-------------------------------------------')
                    print(f"ID: {id_}\nQuestion: \n{question}\n\nOutput: \n{generated_text}\n")
                    print('-------------------------------------------')
                    print(f'Input tokens: {input_length}, Output tokens: {output_length}')
                    print(f'Time: {times / data_num} seconds\n')
                    
                    writer.write({
                        "id": id_,
                        'query': question,
                        "predict": generated_text,
                        "tokens": [input_length, output_length],
                        "time": times / data_num,
                        "note": note,
                    })
        ### 恢复标准输出
        sys.stdout.close()
        sys.stdout = sys.__stdout__
