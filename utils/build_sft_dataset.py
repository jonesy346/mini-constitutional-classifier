import json, random
from pathlib import Path
from typing import List, Dict, Any
import fire

def load_flat_results(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)

def to_sft_records(flat: List[Dict[str, Any]], min_score: int = 4) -> List[Dict[str, Any]]:
    records = []
    for r in flat:
        score = r.get("score", 0) or 0
        if score >= min_score:
            records.append({
                "instruction": r["prompt"].strip(),
                "output": r["candidate"].strip(),
                "meta": {"score": score, "reason": r.get("reason", "").strip()}
            })
    return records

def dedupe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for rec in records:
        key = (rec["instruction"], rec["output"])
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return out

def split(records: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 1337):
    random.seed(seed)
    random.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    return records[n_val:], records[:n_val]

def save_jsonl(records: List[Dict[str, Any]], path: str):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_sft_dataset(flat_path: str = "outputs/filtered_selected_2025_10_10_19_13.json", out_dir="datasets/sft", min_score=4, val_ratio=0.2):
    flat = load_flat_results(flat_path)
    records = to_sft_records(flat, min_score=min_score)
    records = dedupe(records)
    train, val = split(records, val_ratio=val_ratio)
    save_jsonl(train, f"{out_dir}/train.jsonl")
    save_jsonl(val, f"{out_dir}/val.jsonl")
    print(f"✅ SFT dataset built: {len(train)} train / {len(val)} val (min_score={min_score})")

if __name__ == "__main__":
    # Example:
    # python utils/build_sft_dataset.py
    fire.Fire(build_sft_dataset)
