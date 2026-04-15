"""CLI for Indic LLM Eval Harness."""

import click
import yaml
from pathlib import Path
from indic_eval.runner import EvalRunner


@click.group()
def cli():
    """Indic LLM Eval Harness — evaluate LLMs on Indian languages."""
    pass


@cli.command()
@click.option("--model", "-m", required=False, help="Model ID or shortname (e.g. mistral-7b, gemini-flash)")
@click.option("--lang", "-l", required=False, default="hi", show_default=True,
              type=click.Choice(["hi","mr","ta","bn","te","gu","kn","ml","pa","ur"]),
              help="Language code")
@click.option("--tasks", "-t", required=False, default="qa",
              help="Comma-separated tasks: qa, summarization, nli, sentiment, translation")
@click.option("--config", "-c", required=False, type=click.Path(exists=True),
              help="Path to a YAML config file (overrides other flags)")
@click.option("--output", "-o", default="results/", show_default=True, help="Output directory")
@click.option("--num-samples", default=100, show_default=True, help="Number of samples per task")
def evaluate(model, lang, tasks, config, output, num_samples):
    """Run evaluation for a model on Indic language benchmarks."""
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        model = cfg.get("model", model)
        lang = cfg.get("lang", lang)
        tasks = cfg.get("tasks", tasks)
        num_samples = cfg.get("num_samples", num_samples)

    if not model:
        raise click.UsageError("Provide --model or --config.")

    task_list = [t.strip() for t in tasks.split(",")]
    runner = EvalRunner(model=model, lang=lang, tasks=task_list,
                        output_dir=output, num_samples=num_samples)
    runner.run()


@cli.command()
@click.option("--results-dir", default="results/", show_default=True)
def leaderboard(results_dir):
    """Print the leaderboard from saved results."""
    import json, pandas as pd
    lb_path = Path(results_dir) / "leaderboard.json"
    if not lb_path.exists():
        click.echo("No leaderboard found. Run some evaluations first.")
        return
    data = json.loads(lb_path.read_text())
    df = pd.DataFrame(data).sort_values("avg", ascending=False)
    click.echo(df.to_markdown(index=False))


if __name__ == "__main__":
    cli()
