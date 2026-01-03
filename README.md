# Mini Constitutional Classifier

## Introduction

*A lightweight, open-source experiment inspired by Anthropic's Constitutional AI framework.*

This project implements **two complementary pipelines** for building and evaluating safer language models:

### Pipeline 1: Constitutional Response Generation
Generate, critique, and curate AI responses based on constitutional principles. This pipeline helps you build high-quality training datasets by filtering responses according to safety criteria.

### Pipeline 2: Fine-Tuning & Evaluation
Train models on curated data using LoRA (Low-Rank Adaptation), then rigorously evaluate whether fine-tuning actually improved alignment through automated baseline comparisons.

Together, these pipelines enable a complete workflow: from generating safe responses, to training models on them, to validating the improvements.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE 1: Constitutional Response Generation        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Prompts] → [Generator Model] → [Candidate Responses]                  │
│      ↓              ↓                      ↓                            │
│  data/prompts.jsonl  EleutherAI/gpt-neo   Multiple outputs per prompt   │
│                                                                         │
│  [Candidate Responses] → [Critic Model] → [Safety Scores]               │
│       ↓              ↓                  ↓                               │
│  All outputs    google/gemma-2b-it   Score 1-5 + reason                 │
│                                                                         │
│  [Scored Outputs] → [Selection Policy] → [Filtered Dataset]             │
│         ↓                  ↓                      ↓                     │
│  outputs/sample.json   best/margin/diverse   High-quality examples      │
│                                                                         │
│  📁 Key Files:                                                          │
│  • launch_pipeline.py - Main orchestration script                       │
│  • utils/generate_and_critique.py - Core generation & scoring logic     │
│  • utils/selection_policy.py - Filtering strategies                     │
│  • constitution.txt - Safety criteria for evaluation                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE 2: Fine-Tuning & Evaluation                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Filtered Data] → [Build SFT Dataset] → [Train/Val Splits]             │
│         ↓                  ↓                      ↓                     │
│  outputs/filtered_*.json   min_score=4    datasets/sft/*.jsonl          │
│                                                                         │
│  [Train/Val Splits] → [LoRA Fine-Tuning] → [Fine-Tuned Model]           │
│       ↓                        ↓                      ↓                 │
│  train.jsonl          microsoft/phi-2 +     outputs/lora-sft/ (adapter) │
│                         LoRA adapters                                   │
│                                                                         │
│  [Baseline Model] ──┐                                                   │
│  [Fine-Tuned Model] ├→ [Evaluation] → [Metrics & Comparison]            │
│  [Eval Prompts] ────┘       ↓                    ↓                      │
│                    Phase 1: Generate    Mean score, safe rate,          │
│                    Phase 2: Generate    improved fraction,              │
│                    Phase 3: Critique    delta metrics                   │
│                                                                         │
│  📁 Key Files:                                                          │
│  • utils/build_sft_dataset.py - Convert filtered data to SFT format     │
│  • train_lora.py - LoRA fine-tuning script                              │
│  • evaluate_finetune_vs_baseline.py - Rigorous model comparison         │
│  • compare_models.py - Quick side-by-side inference test                │
│  • infer_sft.py - Simple inference with fine-tuned model                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

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

<img width="3072" height="982" alt="image" src="https://github.com/user-attachments/assets/96e74ef1-40a9-48db-ae25-7026f547aaa3" />


**Interpreting Results:**

- **Positive delta_mean**: Fine-tuning improved alignment ✅
- **Negative delta_mean**: Baseline performed better (may need more training data or different hyperparameters)
- **Low improved_fraction with small sample size**: Results may be due to random variance - increase eval set size

**Important Notes:**

- **Small eval sets**: With only 5 prompts, results can be noisy. Consider creating a larger eval set in `datasets/eval/prompts.jsonl`
- **Temperature during eval**: Currently uses `temperature=0.8` which adds randomness. For deterministic evaluation, see configuration options below.
- **Critic model**: Uses `google/gemma-2b-it` for scoring. Ensure you have access to this model on HuggingFace.

### Step 4: Using the Fine-Tuned Model

After training completes, use the fine-tuned model for inference:

```bash
python infer_sft.py
```

This script loads your LoRA adapter from `outputs/lora-sft/` and generates a sample response. Edit the `BASE` variable in `infer_sft.py` to match your training base model, and modify the prompt to test different inputs.

For more advanced usage (side-by-side comparison of baseline vs fine-tuned), see `compare_models.py`.
