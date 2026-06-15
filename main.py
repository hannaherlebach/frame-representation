"""
MMLU abstract algebra evaluation pipeline.

Baseline (no persona):
    python main.py

Specific models + personas:
    python main.py --models claude-sonnet-4-5 \
                   --personas expert_mathematician undergraduate_student

Quick smoke-test (10 questions, no plot):
    python main.py --limit 10 --no-plot
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from evaluate import EvalResult, load_abstract_algebra, run_evaluation
from models import MODEL_CONFIGS
from personas import PERSONAS


def plot_model_results(model_name: str, results: dict[str, EvalResult], output_dir: Path) -> None:
    """Bar chart of accuracy ± std for baseline + each persona for one model."""
    labels = list(results.keys())
    accuracies = [results[k].accuracy * 100 for k in labels]
    stds = [results[k].stderr * 100 for k in labels]

    _, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 6))
    bars = ax.bar(
        labels, accuracies, yerr=stds, capsize=5,
        color="steelblue", edgecolor="white", width=0.5,
        error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
    )

    for bar, acc, std in zip(bars, accuracies, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + std + 2,          # sit just above the error bar cap
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name} — MMLU Abstract Algebra")
    ax.axhline(y=25, color="gray", linestyle="--", linewidth=0.8, label="Random baseline (25%)")
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = output_dir / f"{model_name}_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on MMLU abstract algebra with optional persona prompting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        metavar="MODEL",
        help=f"Models to evaluate. Available: {', '.join(MODEL_CONFIGS)}",
    )
    parser.add_argument(
        "--personas",
        nargs="*",
        choices=list(PERSONAS.keys()),
        metavar="PERSONA",
        help=f"Personas to evaluate (from personas.py). Available: {', '.join(PERSONAS)}",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the no-persona baseline run",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "validation", "dev"],
        help="MMLU dataset split (default: test, 100 questions)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on only the first N questions (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON results and plots (default: results/)",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Chain-of-thought mode: model reasons visibly in output (2048 tokens), "
             "no extended thinking. Mutually exclusive with --thinking-budget.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=1024,
        metavar="TOKENS",
        help="Extended thinking token budget (default: 1024). "
             "Set to 0 to disable thinking and use prefill instead. Claude only.",
    )
    parser.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Optional label appended to the run directory (e.g. 'forbidden_knowledge')",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating accuracy plots",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.name}" if args.name else timestamp
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MMLU abstract_algebra ({args.split})...")
    dataset = load_abstract_algebra(args.split)
    n = args.limit or len(dataset)
    print(f"  {n} questions\n")

    thinking_budget = args.thinking_budget or None  # treat 0 as disabled
    all_results: dict[str, dict[str, EvalResult]] = {}

    for model_name in args.models:
        print(f"── {model_name} ──")
        model_results: dict[str, EvalResult] = {}

        def _run(condition: str, system_prompt) -> EvalResult:
            log_path = run_dir / f"{model_name}__{condition}.jsonl"
            result = run_evaluation(
                model_name, dataset,
                system_prompt=system_prompt,
                thinking_budget=thinking_budget,
                cot=args.cot,
                limit=args.limit,
                log_path=log_path,
            )
            print(f"  {condition}: {result}")
            return result

        if not args.no_baseline:
            model_results["baseline"] = _run("baseline", None)

        for persona_name in args.personas or []:
            model_results[persona_name] = _run(persona_name, PERSONAS[persona_name])

        all_results[model_name] = model_results

        if not args.no_plot and model_results:
            plot_model_results(model_name, model_results, run_dir)

    results_path = run_dir / "results.json"
    serialisable = {
        "_meta": {
            "thinking_budget": thinking_budget,
            "cot": args.cot,
            "split": args.split,
            "limit": args.limit,
            "name": args.name,
        },
        **{
            model: {condition: asdict(r) for condition, r in conditions.items()}
            for model, conditions in all_results.items()
        },
    }
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
