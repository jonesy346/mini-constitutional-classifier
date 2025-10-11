import fire
import json
from tqdm import tqdm
from datetime import datetime

from utils.selection_policy import (
    filter_valid_outputs,
    SELECTION_POLICIES,
)

from utils.evaluation import evaluate_model

from utils.generate_and_critique import (generate_candidates, critic_score, PATH_TO_CONSTITUTION_TXT, PATH_TO_PROMPTS_JSONL)

from utils.normalize import flatten_selection_entries

now = datetime.now()
current_datetime_string = now.strftime("%Y-%m-%d %H:%M")
clean_datetime_string = current_datetime_string.replace(" ", "_").replace(":", "_").replace("-", "_")

PATH_TO_FILTERED_OUTPUTS = f"outputs/filtered_selected_{clean_datetime_string}.json"
PATH_TO_GEN_CRIT_RESULTS = "outputs/gen_crit_results_sample.json"


def run_generation_pipeline():
    constitution = open(PATH_TO_CONSTITUTION_TXT).read()
    prompts = [json.loads(l) for l in open(PATH_TO_PROMPTS_JSONL)]
    results = []
    for prompt in tqdm(prompts, desc="Generating + Critiquing"):
        candidates = generate_candidates(prompt["prompt"], n=2)
        scored = []
        for cand in candidates:
            critique = critic_score(prompt["prompt"], cand, constitution)
            scored.append({
                "candidate": cand.strip(),
                "score": critique["score"],
                "reason": critique["reason"]
            })
        results.append({"id": prompt["id"], "prompt": prompt["prompt"], "scored": scored})
    json.dump(results, open(f"outputs/gen_crit_results_{clean_datetime_string}.json", "w"), indent=2)
    print(f"✅ Saved generation results to {PATH_TO_GEN_CRIT_RESULTS}")
    return results

def run_selection_pipeline(results, policy_name="best"):
    policy = SELECTION_POLICIES[policy_name]
    selected_entries = [res for p in results if (res := policy(p)) is not None]
    print(f"✅ {len(selected_entries)} entries selected via '{policy_name}' policy")

    # optional filter step
    filtered = []
    for entry in selected_entries:
        if "selected" in entry:
            filtered += filter_valid_outputs(entry)
        else:
            filtered.append(entry)
    json.dump(filtered, open(PATH_TO_FILTERED_OUTPUTS, "w"), indent=2)
    print(f"✅ Saved filtered outputs to {PATH_TO_FILTERED_OUTPUTS}")
    return filtered


def launch_pipeline():
    """
    Mini-Constitutional Classifier Pipeline
    --------------------------------------
    1. Generate model candidates and critic scores    
    2. Parse, select, and filter safe outputs         
    3. Evaluate baseline alignment and performance     
    """
    try:
        results = json.load(open(PATH_TO_GEN_CRIT_RESULTS))
        print("Using existing generation results...")
    except FileNotFoundError:
        results = run_generation_pipeline()

    filtered_results = run_selection_pipeline(results, policy_name="margin")
    flat_results = flatten_selection_entries(filtered_results, include="both")
    baseline_metrics = evaluate_model(flat_results)
    print(f"Baseline Safety Evaluation:\n{json.dumps(baseline_metrics, indent=2)}")

if __name__ == "__main__":
    fire.Fire(launch_pipeline)