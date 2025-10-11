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
