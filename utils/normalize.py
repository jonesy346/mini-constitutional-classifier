# utils/normalize.py
from typing import Any

def flatten_selection_entries(entries: list[dict[str, Any]], include: str = "both") -> list[dict[str, Any]]:
    """
    Normalize selection outputs to Day-3 flat schema:
      {"id", "prompt", "candidate", "score", "reason"}

    Parameters
    - entries: list of selection outputs (may come from 'best', 'margin', 'diverse', etc.)
    - include: "both" (default) -> include best and worst if present,
               "best" -> include only best candidates,
               "worst" -> include only worst candidates

    Returns
    - flat: list of dicts in Day-3 format
    """
    flat = []
    for entry in entries:
        entry_id = entry.get("id")
        prompt = entry.get("prompt", "")

        # Case A: Day-2 style (has "scored" list) -> expand each scored item
        if "scored" in entry and isinstance(entry["scored"], list):
            for s in entry["scored"]:
                flat.append({
                    "id": entry_id,
                    "prompt": prompt,
                    "candidate": s.get("candidate", ""),
                    "score": s.get("score"),
                    "reason": s.get("reason", "") or ""
                })
            continue

        # Case B: already Day-3 flat (single candidate)
        if "candidate" in entry and "score" in entry:
            flat.append({
                "id": entry_id,
                "prompt": prompt,
                "candidate": entry.get("candidate", ""),
                "score": entry.get("score"),
                "reason": entry.get("reason", "") or ""
            })
            continue

        # Case C: margin-style with best/worst
        if "best_candidate" in entry and "worst_candidate" in entry:
            if include in ("both", "best"):
                flat.append({
                    "id": entry_id,
                    "prompt": prompt,
                    "candidate": entry.get("best_candidate", ""),
                    "score": entry.get("best_score"),
                    "reason": entry.get("best_reason", "") or ""
                })
            if include in ("both", "worst"):
                flat.append({
                    "id": entry_id,
                    "prompt": prompt,
                    "candidate": entry.get("worst_candidate", ""),
                    "score": entry.get("worst_score"),
                    "reason": entry.get("worst_reason", "") or ""
                })
            continue

        # Case D: diverse-style (has "selected" list)
        if "selected" in entry and isinstance(entry["selected"], list):
            for s in entry["selected"]:
                flat.append({
                    "id": entry_id,
                    "prompt": prompt,
                    "candidate": s.get("candidate", ""),
                    "score": s.get("score"),
                    "reason": s.get("reason", "") or ""
                })
            continue

        # Fallback: attempt to pick any numeric score-like fields
        for key in ("score", "best_score", "worst_score"):
            if key in entry:
                flat.append({
                    "id": entry_id,
                    "prompt": prompt,
                    "candidate": entry.get("candidate") or entry.get("best_candidate") or entry.get("worst_candidate",""),
                    "score": entry.get(key),
                    "reason": entry.get("reason","") or ""
                })
                break

    return flat
