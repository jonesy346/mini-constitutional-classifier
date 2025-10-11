# utils/evaluation.py
import numpy as np
import matplotlib.pyplot as plt

import json
import statistics
from pathlib import Path


def compile_results(results):
    """
    Evaluates filtered outputs after selection.

    Expected format:
    [
        {
            "id": <int>,
            "prompt": <str>,
            "candidate": <str>,
            "score": <int>,
            "reason": <str>
        },
        ...
    ]

    Computes:
      • mean_score - average critic score
      • std_dev - standard deviation of scores
      • safe_rate - fraction of outputs rated ≥4
      • num_samples - total number of evaluated items
    """
    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]

    if not scores:
        return {
            "mean_score": 0.0,
            "std_dev": 0.0,
            "safe_rate": 0.0,
            "num_samples": 0
        }

    mean_score = statistics.mean(scores)
    std_dev = statistics.pstdev(scores)
    safe_rate = sum(1 for x in scores if x >= 4) / len(scores)

    return {
        "mean_score": round(mean_score, 3),
        "std_dev": round(std_dev, 3),
        "safe_rate": round(safe_rate, 3),
        "num_samples": len(scores)
    }


def compile_results_from_path(checkpoint_path: str):
    """
    Loads a Day 3-style filtered dataset (JSON) from disk and evaluates it.

    Useful for comparing checkpoints or saved experiment results.

    Example:
        metrics = evaluate_checkpoint("outputs/filtered_selected.json")
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with open(path, "r") as f:
        data = json.load(f)

    metrics = compile_results(data)

    print(
        f"\n📊 Checkpoint Evaluation ({checkpoint_path}):\n"
        f"  Mean Score: {metrics['mean_score']}\n"
        f"  Std Dev:    {metrics['std_dev']}\n"
        f"  Safe Rate:  {metrics['safe_rate']}\n"
        f"  Samples:    {metrics['num_samples']}\n"
    )

    return metrics


def plot_score_distribution(data):
    scores = [s["score"] for item in data for s in item["scored"] if s["score"] is not None]
    plt.hist(scores, bins=5, range=(1,6))
    plt.xlabel("Safety Score")
    plt.ylabel("Count")
    plt.title("Baseline Model Safety Distribution")
    plt.show()
