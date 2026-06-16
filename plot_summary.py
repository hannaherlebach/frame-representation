"""
Produce a three-subfigure summary plot from reorganised results.

Expected structure:
    results/claude-sonnet-4-5/
        answer_only/   **/results*.json
        cot_thinking_off/  **/results*.json
        thinking_on/   **/results*.json

Each subdirectory may contain multiple run folders; conditions are merged
across them (first occurrence wins for duplicates).

Usage:
    python plot_summary.py [--out summary.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESPONSE_TYPES = {
    "answer_only":      "answer only (thinking off)",
    "cot_thinking_off": "CoT + answer (thinking off)",
    "thinking_on":      "thinking on",
}

# Consistent ordering and display names for conditions
CONDITION_LABELS = {
    "baseline":                "baseline",
    "high_schooler":           "high schooler",
    "maths_undergraduate":     "unexceptional \n maths undergrad",
    "literature_phd":          "literature PhD",
    "literature_phd_explicit": "literature PhD\n+ no maths",
    "literature_phd_extreme":  "literature PhD\n+ bad at maths",
}
CONDITION_ORDER = list(CONDITION_LABELS.keys())

# Literature PhD variants share a colour; others are distinct
_LIT = "#B07BAC"   # muted mauve
CONDITION_COLORS = {
    "baseline":                "#5B8DB8",  # steel blue
    "high_schooler":           "#E09B3D",  # amber
    "maths_undergraduate":     "#5DA86B",  # sage green
    "literature_phd":          _LIT,
    "literature_phd_explicit": _LIT,
    "literature_phd_extreme":  _LIT,
}


def load_response_type(dirpath: Path) -> dict[str, dict]:
    """Aggregate all conditions across run subdirectories; first occurrence wins."""
    conditions: dict[str, dict] = {}
    # Accept both results.json and results_*.json (old format)
    for result_file in sorted(dirpath.rglob("results*.json")):
        with open(result_file) as f:
            data = json.load(f)
        model_data = {k: v for k, v in data.items() if not k.startswith("_")}
        for cond_dict in model_data.values():
            for cond, result in cond_dict.items():
                if cond not in conditions:
                    conditions[cond] = result
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path,
                        default=Path("results/claude-sonnet-4-5"))
    parser.add_argument("--out", type=Path, default=Path("results/summary.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(5, 10), sharey=True)
    fig.suptitle("MMLU abstract algebra accuracy (100 questions)\n by response type and persona: claude-sonnet-4.5",
                 fontsize=13, y=1.02)

    for ax, (dirname, title) in zip(axes, RESPONSE_TYPES.items()):
        dirpath = args.base / dirname
        if not dirpath.exists():
            ax.set_title(f"{title}\n(directory not found)")
            continue

        conditions = load_response_type(dirpath)

        # Use canonical order, skip any conditions not present
        present = [c for c in CONDITION_ORDER if c in conditions]
        missing = [c for c in CONDITION_ORDER if c not in conditions]
        if missing:
            print(f"[{dirname}] missing: {missing}")

        labels  = [CONDITION_LABELS[c] for c in present]
        accs    = [conditions[c]["accuracy"] * 100 for c in present]
        errs    = [conditions[c].get("stderr", 0) * 100 for c in present]

        x = np.arange(len(present))
        colors = [CONDITION_COLORS[c] for c in present]
        bars = ax.bar(x, accs, yerr=errs, capsize=4,
                      color=colors, edgecolor="white", width=0.75,
                      error_kw={"elinewidth": 1.2, "ecolor": "dimgray"})

        for bar, acc, err in zip(bars, accs, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    acc + err + 1.5,
                    f"{acc:.0f}%",
                    ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, 115)
        ax.axhline(25, color="gray", linestyle="--", linewidth=0.8)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy (%)")

    # Shared random-baseline legend entry
    axes[-1].plot([], [], color="gray", linestyle="--", linewidth=0.8,
                  label="Random (25%)")
    axes[-1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
