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

### Step 3: Evaluating Fine-Tuned vs Baseline

After training, you should evaluate whether the fine-tuned model actually improved over the baseline. The evaluation script compares both models on a held-out test set.

#### Running the Evaluation

```bash
python evaluate_finetune_vs_baseline.py \
  --baseline_model_name="google/gemma-2-2b" \
  --finetuned_model_path="outputs/lora-sft"
```

**Parameters:**
- `baseline_model_name`: The base model to compare against (should match your training base model)
- `finetuned_model_path`: Path to your LoRA adapter (default: `outputs/lora-sft`)
- `finetuned_base_model`: (Optional) Override base model for LoRA adapter if different from baseline

**Evaluation Process:**

The script runs in 3 phases to minimize memory usage:
1. **Phase 1**: Generate outputs using the baseline model
2. **Phase 2**: Generate outputs using the fine-tuned model (with LoRA adapter)
3. **Phase 3**: Score all outputs using the critic model (`google/gemma-2b-it`)

**Output Files** (saved to `outputs/eval/`):
- `paired_results.json` - Side-by-side comparison of baseline vs fine-tuned outputs
- `failures.json` - Cases where fine-tuned scored lower than baseline
- `report.json` - Aggregated metrics and summary statistics

#### Understanding the Metrics

The evaluation produces several key metrics:

**Mean Score** (1-5 scale):
- Average safety/alignment score from the critic model
- Higher is better (more aligned with constitution)
- Example: `baseline: 4.2, finetuned: 4.0`

**Standard Deviation**:
- Measures consistency of scores across prompts
- Lower variance = more predictable behavior
- Example: `baseline: 0.4, finetuned: 0.0` (all responses got same score)

**Safe Rate** (0-1):
- Fraction of responses scoring ≥ 4 (the safety threshold)
- Target: 1.0 (100% safe responses)
- Example: `baseline: 1.0, finetuned: 1.0` (both perfectly safe)

**Improved Fraction** (0-1):
- Fraction of prompts where fine-tuned scored higher than baseline
- Target: > 0.5 (fine-tuned wins majority of comparisons)
- Example: `0.0` means baseline won every comparison

**Delta Metrics**:
- `delta_mean`: Difference in average scores (positive = fine-tuned better)
- `delta_safe_rate`: Difference in safety rates

#### Example Output

```json
{
  "baseline": {
    "mean_score": 4.2,
    "std_dev": 0.4,
    "safe_rate": 1.0,
    "num_samples": 5
  },
  "finetuned": {
    "mean_score": 4.0,
    "std_dev": 0.0,
    "safe_rate": 1.0,
    "num_samples": 5
  },
  "delta": {
    "delta_mean": -0.2,
    "delta_safe_rate": 0.0,
    "improved_fraction": 0.0
  }
}
```

**Interpreting Results:**

- **Positive delta_mean**: Fine-tuning improved alignment ✅
- **Negative delta_mean**: Baseline performed better (may need more training data or different hyperparameters)
- **Low improved_fraction with small sample size**: Results may be due to random variance - increase eval set size

**Important Notes:**

- **Small eval sets**: With only 5 prompts, results can be noisy. Consider creating a larger eval set in `datasets/eval/prompts.jsonl`
- **Temperature during eval**: Currently uses `temperature=0.8` which adds randomness. For deterministic evaluation, see configuration options below.
- **Critic model**: Uses `google/gemma-2b-it` for scoring. Ensure you have access to this model on HuggingFace.

### Step 4: Using the Fine-Tuned Model

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
