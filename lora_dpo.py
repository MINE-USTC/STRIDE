import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import argparse
import torch
import jsonlines
import swanlab
from swanlab.integration.transformers import SwanLabCallback

from stride.paths import default_ft_dpo_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Base or merged SFT checkpoint (Hugging Face id or local path) for DPO",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="jsonl with DPO rows: prompt, chosen, rejected",
    )
    parser.add_argument(
        "--data_mode", type=str, default="v1", help="Suffix for SwanLab run name"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-6, help="learning rate"
    )
    parser.add_argument(
        '--epoch', type=int, default=1, help="training epoch"
    )
    parser.add_argument(
        '--save_strategy', type=str, default='steps', help="save strategy"
    )
    parser.add_argument(
        '--save_steps', type=int, default=None, help="save steps"
    )
    parser.add_argument(
        '--output_dir', type=str, default=default_ft_dpo_output(), help="DPO checkpoint directory"
    )
    parser.add_argument(
        '--lora_rank', type=int, default=8, help="lora rank"
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=32, help="lora alpha"
    )
    parser.add_argument(
        '--beta', type=float, default=0.1, help="DPO beta"
    )
    parser.add_argument(
        '--run_name', type=str, default='Qwen3-8B-Lora-DPO',
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side='right',
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
    )
    print("Config:", config)

    with jsonlines.open(args.data_path, 'r') as f:
        datas = [obj for obj in f]

    dpo_dataset = Dataset.from_list(datas)

    epochs = args.epoch
    batch_size = args.batch_size
    grad_acc_steps = 16
    total_steps = (len(dpo_dataset) * epochs) // (batch_size * grad_acc_steps)
    steps_per_epoch = math.ceil(len(dpo_dataset) / (batch_size * grad_acc_steps))
    print(f"Total training steps: {total_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    save_steps = max(1, steps_per_epoch // 4)
    print(f"Model will be saved every {save_steps} steps.")

    training_args = DPOConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        save_strategy=args.save_strategy,
        save_steps=save_steps,
        gradient_checkpointing=False,
        beta=args.beta,
        max_length=1300,
        max_prompt_length=1000,
        output_dir=args.output_dir,
        logging_steps=max(2, steps_per_epoch // 5),
        report_to="swanlab",
        run_name=args.run_name + '-' + args.data_mode,
    )

    exp_name = f"{args.run_name}-{args.data_mode}-{len(dpo_dataset)}-lr{args.lr}-epoch{args.epoch}-r{args.lora_rank}-a{args.lora_alpha}-b{args.beta}"

    swanlab_callback = SwanLabCallback(
        project="Qwen3-Lora-DPO",
        experiment_name=exp_name,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        peft_config=config,
        train_dataset=dpo_dataset,
        callbacks=[swanlab_callback]
    )
    print("Start Training!")

    trainer.train()
    swanlab.finish()
