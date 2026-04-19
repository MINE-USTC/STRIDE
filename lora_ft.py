import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer
import argparse
import swanlab
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model

from stride.paths import default_ft_reasoner_output


MAX_LENGTH = 1024


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=8, type=int, help="batch size"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Base causal LM (Hugging Face id or local path) to attach LoRA adapters to",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Training data: Hugging Face Dataset directory from stride.ft_preprocess (save_to_disk)",
    )
    parser.add_argument(
        "--data_mode", type=str, default='v1', help="Suffix for SwanLab run name"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        '--epoch', type=int, default=5, help="training epoch"
    )
    parser.add_argument(
        '--save_strategy', type=str, default='epoch', help="save strategy"
    )
    parser.add_argument(
        '--save_steps', type=int, default=None, help="save steps"
    )
    parser.add_argument(
        '--output_dir', type=str, default=default_ft_reasoner_output(), help="LoRA checkpoint directory"
    )
    parser.add_argument(
        '--lora_rank', type=int, default=8, help="lora rank"
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=32, help="lora alpha"
    )
    parser.add_argument(
        '--run_name', type=str, default='Qwen3-8B-Lora',
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1
    )
    print("Config:", config)
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    train_dataset = load_from_disk(args.data_path)

    train_args = TrainingArguments(
        output_dir=args.output_dir, 
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=args.epoch,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        learning_rate=args.lr,
        gradient_checkpointing=False,
        report_to="swanlab",
        run_name=args.run_name + '-' + args.data_mode,
    )
    
    swanlab_callback = SwanLabCallback(
        project="Qwen3-Lora", 
        experiment_name=args.run_name + '-' + args.data_mode,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback]
    )
    
    print("Start Training!")

    trainer.train()

    swanlab.finish()
