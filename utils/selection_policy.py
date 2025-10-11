# utils/selection_policy.py
def select_best_candidate(prompt_entry):
    """
    Best-Answer Policy
    ------------------
    Selects the single highest-scoring candidate per prompt entry.

    Context:
    In supervised fine-tuning (SFT) or alignment tasks, we often train models to imitate
    the best-rated, safest, or most aligned response. This policy extracts that example.
    """
    scored = [s for s in prompt_entry["scored"] if s["score"] is not None]
    if not scored:
        return None
    best = max(scored, key=lambda x: x["score"])
    return {
        "id": prompt_entry["id"],
        "prompt": prompt_entry["prompt"],
        "candidate": best["candidate"],
        "score": best["score"],
        "reason": best["reason"]
    }


def select_margin_candidates(prompt_entry, margin=2):
    """
    Margin Policy
    -------------
    Keeps only prompts where the difference between the best and worst candidate scores
    exceeds a defined margin (default: 2). Returns both best and worst examples.

    Context:
    Used in reward modeling or pairwise preference learning (e.g., RLHF).
    Strong score gaps create clearer preference signals, improving reward model stability.
    """
    scored = [s for s in prompt_entry["scored"] if s["score"] is not None]
    if len(scored) < 2:
        return None

    best = max(scored, key=lambda x: x["score"])
    worst = min(scored, key=lambda x: x["score"])

    if best["score"] - worst["score"] >= margin:
        return {
            "id": prompt_entry["id"],
            "prompt": prompt_entry["prompt"],
            "best_candidate": best["candidate"],
            "best_score": best["score"],
            "worst_candidate": worst["candidate"],
            "worst_score": worst["score"]
        }
    return None


def select_diverse_candidates(prompt_entry, top_k=1, bottom_k=1):
    """
    Diversity Policy
    ----------------
    Selects both high- and low-scoring candidates per prompt, up to top_k and bottom_k.

    Context:
    In contrastive training or dataset curation, exposing models to both good and bad
    examples encourages robust alignment and discourages harmful or low-quality outputs.
    """
    scored_list = [scored for scored in prompt_entry["scored"] if scored["score"] is not None]
    if not scored_list:
        return None

    sorted_scores = sorted(scored_list, key=lambda x: x["score"], reverse=True)
    top = sorted_scores[:top_k]
    bottom = sorted_scores[-bottom_k:] if len(sorted_scores) > bottom_k else []

    return {
        "id": prompt_entry["id"],
        "prompt": prompt_entry["prompt"],
        "selected": top + bottom
    }

SELECTION_POLICIES = {
    "best": select_best_candidate,
    "margin": select_margin_candidates,
    "diverse": select_diverse_candidates,
}

# Example usage
POLICY = SELECTION_POLICIES["margin"]
SAFE_THRESHOLD = 4

def select_outputs(data, policy = select_best_candidate):
    return [res for prompt in data if (res := policy(prompt)) is not None]

def filter_valid_outputs(entry, safe_threshold=SAFE_THRESHOLD):
    """
    Filters out logically inconsistent candidates.
    Removes samples where:
      - The critic rated them as safe (score > safe_threshold)
      - But the reason text still mentions unsafe behavior.

    Parameters
    ----------
    entry : dict
        A selected prompt entry, e.g. from select_diverse_candidates().
    safe_threshold : int, optional
        Score cutoff for "safe" examples. Default = SAFE_THRESHOLD.

    Returns
    -------
    list[dict]
        List of valid candidate dicts that pass the filter.
    """
    valid = []
    for candidate in entry.get("selected", []):
        score = candidate.get("score")
        reason = candidate.get("reason", "").lower()
        # mark as invalid if critic contradicts itself
        contradiction = score and score > safe_threshold and "unsafe" in reason
        if not contradiction:
            valid.append(candidate)
    return valid
