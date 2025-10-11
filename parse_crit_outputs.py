import json
from statistics import mean

PATH_TO_CRIT_RESULTS_JSON = "outputs/gen_crit_results_sample.json"

if __name__ == "__main__":
    with open(PATH_TO_CRIT_RESULTS_JSON) as f:
        data = json.load(f)

    # --- Basic parsing + verification ---
    for item in data:
        for metric in item["scored"]:
            if isinstance(metric.get("score"), str):
                try:
                    metric["score"] = int(metric["score"])
                except:
                    metric["score"] = None

    # --- Basic stats ---
    all_scores = [s["score"] for item in data for s in item["scored"] if s["score"] is not None]
    print(f"✅ Parsed {len(all_scores)} valid scores")
    print(f"Mean safety score: {mean(all_scores):.2f}")
    print(f"Score distribution: { {i: all_scores.count(i) for i in range(1,6)} }")
