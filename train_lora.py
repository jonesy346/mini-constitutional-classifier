import os, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

@dataclass
class TrainArgs:
    # base_model: str = "microsoft/phi-2"  # phi-2 is better for reasoning/instruction tasks
    # base_model: str = "gpt2"  # GPT-2 (124M) - lightweight option
    base_model: str = "google/gemma-2b-it"  # Gemma 2B instruct
    train_path: str = "datasets/sft/train.jsonl"
    val_path: str = "datasets/sft/val.jsonl"
    out_dir: str = "outputs/lora-sft"
    use_4bit: bool = False     # True requires CUDA + bitsandbytes
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lr: float = 2e-4
    epochs: int = 3
    bs: int = 4
    grad_accum: int = 4
    max_seq_len: int = 512
    seed: int = 1337

def load_model_and_tokenizer(model_name: str, use_4bit: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if use_4bit:
        # CUDA-only QLoRA path
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    return model, tok

def format_example(example):
    # Map your JSONL fields to a plain instruction-tune text
    instr = example["instruction"].strip()
    out = example["output"].strip()
    # Keep it simple; no special role tokens needed for generic models
    return f"Instruction:\n{instr}\n\nResponse:\n{out}"

def main(cfg: TrainArgs):
    """
    Goal (what we'll have by the end):
    - A clean SFT dataset (JSONL) derived from your filtered selections.
	- A LoRA training script (two variants: local M1-friendly small model, and GPU/Colab A100 for bigger models).
	- Saved LoRA adapter (and optional merged weights) + a tiny inference script to sanity-check the fine-tune.
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    train_ds = load_dataset("json", data_files=cfg.train_path, split="train")
    val_ds = load_dataset("json", data_files=cfg.val_path, split="train")
    train_ds = train_ds.map(lambda ex: {"text": format_example(ex)})
    val_ds = val_ds.map(lambda ex: {"text": format_example(ex)})

    model, tok = load_model_and_tokenizer(cfg.base_model, cfg.use_4bit)

    # LoRA config
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "k_proj", "v_proj", "dense"]  # Phi-2 module names
        # target_modules=["c_attn", "c_proj"]  # GPT-2 module names
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Gemma/LLaMA-style module names
    )
    model = get_peft_model(model, lora_cfg)

    train_args = SFTConfig(
        output_dir=cfg.out_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.bs,
        per_device_eval_batch_size=cfg.bs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        logging_steps=10,
        eval_strategy="steps",  # renamed from evaluation_strategy in newer TRL
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        seed=cfg.seed,
        bf16=torch.cuda.is_available(),  # safe on A100; ignored on MPS/CPU
        fp16=False,
        report_to=[],
        dataset_text_field="text",
        max_length=cfg.max_seq_len,  # renamed from max_seq_length in newer TRL
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=train_args,
    )

    trainer.train()
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"✅ Saved LoRA adapter to {cfg.out_dir}")

if __name__ == "__main__":
    # quick CLI-free run
    cfg = TrainArgs()
    main(cfg)
