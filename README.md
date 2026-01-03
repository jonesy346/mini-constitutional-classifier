# Mini Constitutional Classifier

## Introduction

*A lightweight, open-source experiment inspired by Anthropic’s Constitutional AI framework.*

This project implements an end-to-end **LLM inference and safety evaluation pipeline**, allowing you to generate, critique, and select AI model responses based on human-aligned criteria.

## Overview

The pipeline automates four core stages:

1. **Generation** — Produce multiple candidate responses from a base language model to a set of arbitrary prompts (either adversarial, borderline, or cooperating).
2. **Critique** — Score each candidate via a judge/“critic” model using safety and alignment prompts.
3. **Selection** — Apply configurable policies (best, margin, or diverse) to curate safe, high-quality samples.
4. **Evaluation** — Compile safety metrics and performance summaries for baseline comparison.

## Features

- **Multi-model integration** — plug-and-play support for models like:
  - `EleutherAI/gpt-neo-125M` (fast baseline)
  - `microsoft/phi-2`
  - `google/gemma-2b-it`
- **Configurable critic** — choose between instruct or non-instruct styles.
- **Selection policies**
  - *Best*: choose top-rated responses.
  - *Margin*: keep examples with strong preference gaps.
  - *Diverse*: retain both good and bad outputs for contrastive learning.
- **Evaluation metrics**
  - Mean score, safety rate, and sample-level variance.
- **Hardware optimized**
  - Supports `MPS` (Apple Silicon) and CPU backends with low memory mode.

## Tech Setup
To start, setup your VS Code environment with the following commands:
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Sample Command
Run the pipeline with a command like:
```
python launch_pipeline.py \
--critic_model google/gemma-2b-it \
--policy_name best \
--path_to_gen_crit_results outputs/sample.json
```

This generates responses to prompts from `prompts.json` using the EleutherAI/gpt-neo-125M model then judges them as unsafe/safe (according to the constitution) using the Google gemma model from HuggingFace (authentication required to use) - generations w/ critiques are stored in `outputs/sample.json`. Finally, results are compiled the "best" policy and displayed to the console.

<img width="1717" height="508" alt="Screenshot 2025-10-10 at 11 06 41 PM" src="https://github.com/user-attachments/assets/9c2e9d70-2111-47a1-8039-eab3823c6823" />

## Training

After generating and critiquing responses, you can fine-tune a model using LoRA (Low-Rank Adaptation) on your curated dataset.

### Step 1: Build SFT Dataset

Convert your filtered results into a supervised fine-tuning (SFT) dataset:

```bash
python utils/build_sft_dataset.py \
  --flat_path outputs/filtered_selected_2025_10_10_19_13.json \
  --out_dir datasets/sft \
  --min_score 4 \
  --val_ratio 0.2
```

**Parameters:**
- `flat_path`: Path to your filtered/selected results from the pipeline
- `out_dir`: Output directory for train/val JSONL files
- `min_score`: Minimum critique score to include (1-5 scale)
- `val_ratio`: Fraction of data to use for validation (default: 0.2)

This creates:
- `datasets/sft/train.jsonl` - Training examples
- `datasets/sft/val.jsonl` - Validation examples

### Step 2: Run LoRA Fine-Tuning

Fine-tune a model using the prepared dataset:

```bash
python train_lora.py
```

**Default configuration** (in `train_lora.py`):
- **Model**: `microsoft/phi-2` (2.7B parameters)
- **Method**: LoRA adapters on attention layers
- **Training**: 3 epochs with cosine learning rate schedule
- **Output**: Saved to `outputs/lora-sft/`

**Switching models:**

To use a different base model, edit the `TrainArgs` in `train_lora.py`:

```python
# For GPT-2 (lightweight, fast)
base_model: str = "gpt2"
target_modules=["c_attn", "c_proj"]

# For Gemma-2B (instruct-tuned)
base_model: str = "google/gemma-2b-it"
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# For Phi-2 (recommended, better reasoning)
base_model: str = "microsoft/phi-2"
target_modules=["q_proj", "k_proj", "v_proj", "dense"]
```

**Training hyperparameters:**
- `lora_r`: LoRA rank (default: 8)
- `lora_alpha`: LoRA scaling (default: 16)
- `lr`: Learning rate (default: 2e-4)
- `epochs`: Training epochs (default: 3)
- `bs`: Batch size per device (default: 4)
- `max_seq_len`: Maximum sequence length (default: 512)

**Hardware notes:**
- **M1/M2 Mac**: Uses MPS acceleration automatically
- **CUDA GPU**: Set `use_4bit=True` for QLoRA (requires bitsandbytes)
- **CPU**: Works but slower

### Step 3: Using the Fine-Tuned Model

After training completes, the LoRA adapter and tokenizer are saved to `outputs/lora-sft/`. To use the fine-tuned model for inference, load both the base model and the adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "outputs/lora-sft")

# Generate text
prompt = "Instruction:\nExplain what model alignment means in simple terms.\n\nResponse:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```
