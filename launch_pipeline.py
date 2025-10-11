from pathlib import Path
import fire
import json
from tqdm import tqdm
from datetime import datetime

from utils.selection_policy import (
    filter_valid_outputs,
    SELECTION_POLICIES,
)

from utils.models import get_models

from utils.evaluation import compile_results

from utils.generate_and_critique import (generate_candidates, critic_score, PATH_TO_CONSTITUTION_TXT, PATH_TO_PROMPTS_JSONL)

from utils.normalize import flatten_selection_entries

now = datetime.now()
current_datetime_string = now.strftime("%Y-%m-%d %H:%M")
clean_datetime_string = current_datetime_string.replace(" ", "_").replace(":", "_").replace("-", "_")

PATH_TO_FILTERED_OUTPUTS = f"outputs/filtered_selected_{clean_datetime_string}.json"
PATH_TO_GEN_CRIT_RESULTS = "outputs/gen_crit_results_sample.json"


# -------------------------------------------
# Pipeline Entry Point
# -------------------------------------------
def run_generation_pipeline(models, path_to_prompts_jsonl, path_to_constitution_txt, output_path, num_gen_candidates=2):
    """
    Runs candidate generation and scoring using provided generator + critic models.
    """
    constitution = open(path_to_constitution_txt).read()
    prompts = [json.loads(l) for l in open(path_to_prompts_jsonl)]
    results = []

    gen_model = models["gen_model"]
    gen_tok = models["gen_tok"]
    crit_model = models["crit_model"]
    crit_tok = models["crit_tok"]
    device = models["device"]

    for prompt in tqdm(prompts, desc="Generating + Critiquing"):
        candidates = generate_candidates(prompt["prompt"], gen_model, gen_tok, device, n=num_gen_candidates)
        scored = []
        for cand in candidates:
            critique = critic_score(prompt["prompt"], cand, constitution, crit_model, crit_tok, device)
            scored.append({
                "candidate": cand.strip(),
                "score": critique["score"],
                "reason": critique["reason"]
            })
        results.append({"id": prompt["id"], "prompt": prompt["prompt"], "scored": scored})

    json.dump(results, open(output_path, "w"), indent=2, ensure_ascii=False)
    print(f"✅ Saved generation results to {output_path}")
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


def launch_pipeline(
    gen_model: str = "EleutherAI/gpt-neo-125M",
    critic_model: str = "EleutherAI/gpt-neo-125M",
    policy_name: str = "margin",
    include: str = "both",
    num_gen_candidates: int = 2,
    regenerate: bool = False,
    path_to_prompts: str = PATH_TO_PROMPTS_JSONL,
    path_to_gen_crit_results: str = PATH_TO_GEN_CRIT_RESULTS,
):
    """
    Mini-Constitutional Classifier Pipeline
    --------------------------------------
    1. Generate model candidates and critic scores    
    2. Parse, select, and filter safe outputs         
    3. Evaluate baseline alignment and performance     

    Parameters
    ----------
    critic_model : str
        The name of the Hugging Face model to use as the critic.
        Options include:
            - "EleutherAI/gpt-neo-125M" (lightweight baseline)
            - "microsoft/phi-2"         (strong, open-access)
            - "google/gemma-2b-it"      (instruct, gated access)
    policy_name : str
        The selection policy to use ("best", "margin", or "diverse").
    include : str
        Whether to include "best", "worst", or "both" candidates during flattening.
    num_gen_candidates : int
        The number of candidates to generate per prompt.
    regenerate : bool
        If True, forces regeneration even if cached results exist.
    """

    print(f"\n🚀 Launching pipeline with:")
    print(f"  Gen Model:    {gen_model}")
    print(f"  Critic Model: {critic_model}")
    print(f"  Policy:       {policy_name}")
    print(f"  Include:      {include}")
    print(f"  Candidates:   {num_gen_candidates}")
    print(f"  Regenerate:   {regenerate}\n")

    # --- Step 1: Run or load generation results ---
    path = Path(path_to_gen_crit_results)
    if not regenerate and path.exists():
        print("Using existing generation results...")
        results = json.load(open(path_to_gen_crit_results))
    else:
        print("Generating new candidate + critic results...")
        models = get_models(gen_model, critic_model)
        results = run_generation_pipeline(
            models,
            path_to_prompts,
            PATH_TO_CONSTITUTION_TXT,
            path_to_gen_crit_results,
            num_gen_candidates=num_gen_candidates
        )
        with open(path, "w") as fw:
            json.dump(results, fw, indent=2, ensure_ascii=False)

      # --- Step 2: Apply selection policy and flatten ---
    filtered_results = run_selection_pipeline(results, policy_name=policy_name)
    flat_results = flatten_selection_entries(filtered_results, include=include)

    # --- Step 3: Evaluate baseline ---
    baseline_metrics = compile_results(flat_results)
    print(f"\n📊 Baseline Safety Evaluation:\n{json.dumps(baseline_metrics, indent=2)}")

    return baseline_metrics

if __name__ == "__main__":
    fire.Fire(launch_pipeline)