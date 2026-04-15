"""Core evaluation runner — orchestrates dataset loading, model inference, and scoring."""

import json
import time
from pathlib import Path
from tqdm import tqdm

from indic_eval.datasets.loader import DatasetLoader
from indic_eval.models.factory import get_model
from indic_eval.metrics.factory import get_metric
from indic_eval.tasks.registry import TASK_REGISTRY


class EvalRunner:
    def __init__(self, model: str, lang: str, tasks: list[str],
                 output_dir: str = "results/", num_samples: int = 100):
        self.model_name = model
        self.lang = lang
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        print(f"\n🚀 Indic LLM Eval Harness")
        print(f"   Model : {self.model_name}")
        print(f"   Lang  : {self.lang}")
        print(f"   Tasks : {', '.join(self.tasks)}\n")

        model = get_model(self.model_name)
        all_results = {}

        for task_name in self.tasks:
            if task_name not in TASK_REGISTRY:
                print(f"⚠️  Unknown task '{task_name}', skipping.")
                continue

            task_cfg = TASK_REGISTRY[task_name]
            print(f"📋 Task: {task_name}")

            loader = DatasetLoader(task_cfg["dataset"], self.lang, self.num_samples)
            samples = loader.load()

            if not samples:
                print(f"   No data found for {task_name}/{self.lang}, skipping.\n")
                continue

            predictions, references = [], []
            for sample in tqdm(samples, desc=f"   Running {task_name}", ncols=70):
                prompt = task_cfg["prompt_fn"](sample, self.lang)
                pred = model.generate(prompt)
                predictions.append(pred)
                references.append(sample["answer"])

            metric_fn = get_metric(task_cfg["metric"])
            score = metric_fn(predictions, references)
            all_results[task_name] = score
            print(f"   ✅ {task_cfg['metric']}: {score}\n")

        self._save_results(all_results)
        self._update_leaderboard(all_results)
        print(f"📁 Results saved to {self.output_dir}/")

    def _save_results(self, results: dict):
        ts = int(time.time())
        out = {
            "model": self.model_name,
            "lang": self.lang,
            "timestamp": ts,
            "results": results,
        }
        path = self.output_dir / f"{self.model_name.replace('/', '_')}_{self.lang}_{ts}.json"
        path.write_text(json.dumps(out, indent=2))

    def _update_leaderboard(self, results: dict):
        lb_path = self.output_dir / "leaderboard.json"
        leaderboard = json.loads(lb_path.read_text()) if lb_path.exists() else []

        scores = [v for v in results.values() if isinstance(v, (int, float))]
        avg = round(sum(scores) / len(scores), 2) if scores else 0

        entry = {"model": self.model_name, "lang": self.lang, "avg": avg, **results}
        leaderboard = [r for r in leaderboard
                       if not (r["model"] == self.model_name and r["lang"] == self.lang)]
        leaderboard.append(entry)
        lb_path.write_text(json.dumps(leaderboard, indent=2))
