# Mini Constitutional Classifier

### Introduction
This is a reproducible pipeline that: 
1. Generates N candidate responses to a set of arbitrary prompts (either adversarial, borderline, or cooperating)
2. Judges/critiques each candidate vs a constitution
3. Picks (or re-ranks) the best answer
4. Produces evaluation metrics (harmlessness, helpfulness proxy)

### Tech Setup
To start, setup your VS Code environment with the following commands:
```
python -m venv venv && source venv/bin/activate
pip install -U pip
pip install transformers datasets peft accelerate bitsandbytes evaluate streamlit flask sentence-transformers matplotlib fire
```

### Sample Command
Run the pipeline with a command like:
```
python launch_pipeline.py \
--critic_model google/gemma-2b-it \
--policy_name best \
--path_to_gen_crit_results outputs/sample.json
```

This generates responses to prompts from prompts.json using the EleutherAI/gpt-neo-125M model then judges them as unsafe/safe (according to the constitution) using the Google gemma model from HuggingFace (authentication required to use). Finally, results are compiled the "best" policy and displayed to the console.

<img width="1717" height="508" alt="Screenshot 2025-10-10 at 11 06 41 PM" src="https://github.com/user-attachments/assets/9c2e9d70-2111-47a1-8039-eab3823c6823" />
