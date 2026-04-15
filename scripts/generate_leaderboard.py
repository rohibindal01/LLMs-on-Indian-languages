#!/usr/bin/env python3
"""
scripts/generate_leaderboard.py
--------------------------------
Reads all result JSON files in results/ and rebuilds leaderboard.json.
Run after manually adding result files:
    python scripts/generate_leaderboard.py
"""

import json
from pathlib import Path


RESULTS_DIR = Path("results")


def load_results() -> list[dict]:
    entries = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "leaderboard.json":
            continue
        try:
            data = json.loads(path.read_text())
            scores = data.get("results", {})
            flat_scores = {}
            for task, val in scores.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        flat_scores[f"{task}_{k}"] = v
                else:
                    flat_scores[task] = val
            numeric = [v for v in flat_scores.values() if isinstance(v, (int, float))]
            avg = round(sum(numeric) / len(numeric), 2) if numeric else 0
            entry = {"model": data["model"], "lang": data["lang"], "avg": avg, **flat_scores}
            entries.append(entry)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
    return entries


def write_leaderboard(entries: list[dict]):
    lb_path = RESULTS_DIR / "leaderboard.json"
    seen = {}
    for e in entries:
        key = (e["model"], e["lang"])
        if key not in seen or e["avg"] > seen[key]["avg"]:
            seen[key] = e
    leaderboard = sorted(seen.values(), key=lambda x: x["avg"], reverse=True)
    lb_path.write_text(json.dumps(leaderboard, indent=2))
    print(f"✅ Leaderboard written: {len(leaderboard)} entries → {lb_path}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    entries = load_results()
    write_leaderboard(entries)
