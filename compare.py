"""
Plot accuracy side-by-side across multiple experiment runs.

Each run directory must contain a results.json produced by main.py.
Bars are grouped by condition (persona), coloured by mode (thinking / cot / prefill).

Usage:
    # Compare two runs across all shared conditions
    python compare.py results/20260615_thinking results/20260615_cot

    # Restrict to specific conditions
    python compare.py results/run1 results/run2 \
        --conditions baseline literature_phd_explicit

    # Custom output path and title
    python compare.py results/run1 results/run2 \
        --out figures/comparison.png --title "Thinking vs CoT"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = ["steelblue", "tomato", "mediumseagreen", "mediumpurple", "darkorange"]


def load_run(run_dir: Path) -> dict:
    path = run_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"No results.json found in {run_dir}")
    with open(path) as f:
        return json.load(f)


def mode_label(meta: dict) -> str:
    """Derive a short mode string from run metadata."""
    if meta.get("cot"):
        return "cot"
    budget = meta.get("thinking_budget")
    if budget:
        return f"thinking-{budget}"
    return "prefill"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MMLU accuracy across experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("runs", nargs="+", type=Path, metavar="RUN_DIR",
                        help="Result directories to compare (must contain results.json)")
    parser.add_argument("--conditions", nargs="*", default=None, metavar="CONDITION",
                        help="Conditions to include (default: all shared conditions). "
                             "E.g. baseline literature_phd_explicit")
    parser.add_argument("--out", type=Path, default=Path("comparison.png"),
                        help="Output plot path (default: comparison.png)")
    parser.add_argument("--title", default="MMLU Abstract Algebra — Run Comparison")
    args = parser.parse_args()

    # Load each run and determine its mode label
    runs = []
    for run_dir in args.runs:
        data = load_run(run_dir)
        meta = data.get("_meta", {})
        # Model results are all top-level keys that aren't "_meta"
        model_results = {k: v for k, v in data.items() if not k.startswith("_")}
        runs.append({
            "dir": run_dir,
            "mode": mode_label(meta),
            "results": model_results,
        })

    # Collect conditions present in every run (intersection), preserving order
    all_condition_sets = [
        list(conds.keys())
        for run in runs
        for conds in run["results"].values()
    ]
    # Union of all conditions seen, in first-seen order
    seen: dict[str, None] = {}
    for cond_list in all_condition_sets:
        for c in cond_list:
            seen[c] = None
    all_conditions = list(seen)

    conditions = [c for c in all_conditions if c in (args.conditions or all_conditions)]
    if not conditions:
        raise ValueError("No matching conditions found across the selected runs.")

    # Assign a colour to each mode
    unique_modes = list(dict.fromkeys(r["mode"] for r in runs))
    mode_color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(unique_modes)}

    n_conds = len(conditions)
    n_runs = len(runs)
    bar_width = 0.7 / n_runs
    x = np.arange(n_conds)

    _, ax = plt.subplots(figsize=(max(7, n_conds * n_runs * 0.9), 6))

    for i, run in enumerate(runs):
        mode = run["mode"]
        accs, errs = [], []
        for cond in conditions:
            # Take results from whichever model is in the run (assumes one model per run)
            model_data = next(iter(run["results"].values()), {})
            entry = model_data.get(cond)
            accs.append(entry["accuracy"] * 100 if entry else 0.0)
            errs.append(entry.get("stderr", 0) * 100 if entry else 0.0)

        offset = (i - n_runs / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, accs, bar_width,
            yerr=errs, capsize=4,
            color=mode_color[mode], edgecolor="white",
            error_kw={"elinewidth": 1.2, "ecolor": "dimgray"},
            label=mode,
        )
        for bar, acc, err in zip(bars, accs, errs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    acc + err + 1.5,
                    f"{acc:.0f}%",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(args.title)
    ax.axhline(y=25, color="gray", linestyle="--", linewidth=0.8)

    legend_handles = [
        mpatches.Patch(color=mode_color[m], label=m) for m in unique_modes
    ] + [plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8, label="Random (25%)")]
    ax.legend(handles=legend_handles, fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
