"""Regenerate the figures used in the README.

The numbers below are the measured results reported in the internship report
(see the evaluation scripts in addition/eval and sorting/eval).

Usage:  python figures/make_figures.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, RED, GREY = "#1f5fa8", "#e8890c", "#b91d47", "#555555"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})


def addition_data_scaling():
    """3-digit addition: accuracy vs. number of scratchpad training examples."""
    n_examples = [0, 100, 250, 500, 1000, 2000]
    accuracy = [0, 5, 20, 48, 75, 93]  # % correct on 200 random 3-digit additions
    total_pairs = 900 * 900  # all (x, y) pairs with 100 <= x, y <= 999

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    ax.plot(n_examples, accuracy, marker="o", color=BLUE, linewidth=2.5, markersize=7)
    for x, y in zip(n_examples, accuracy):
        ax.annotate(f"{y}%", (x, y), textcoords="offset points",
                    xytext=(0, 9) if x < 2000 else (-22, -4), ha="center", fontweight="bold")
    ax.set_xticks(n_examples)
    ax.set_xlabel("Number of 3-digit training examples")
    ax.annotate(f"2,000 examples = {100 * 2000 / total_pairs:.2f}% of the\n810,000 possible 3-digit pairs",
                (2000, 93), textcoords="offset points", xytext=(-150, -60), fontsize=9, color=GREY,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_ylabel("Accuracy on 200 unseen additions (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("GPT-2 (124M) on 3-digit addition with a step-by-step scratchpad",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "addition_data_scaling.png"))
    plt.close(fig)


def sorting_items_lost():
    """Sorting: average number of input items missing from the final answer."""
    lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    lost = [0, 0, 0, 0, 0, 1.0, 1.8, 2.5, 3.2]  # mean over 10 lists per length
    colors = [BLUE if L <= 5 else (GREY if L == 6 else ORANGE if L < 10 else RED) for L in lengths]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    bars = ax.bar(lengths, lost, color=colors, width=0.65)
    for b, v in zip(bars, lost):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontweight="bold")
    ax.axvspan(1.5, 5.5, color=BLUE, alpha=0.06)
    ax.text(3.5, 3.3, "training lengths (3-5)", ha="center", color=BLUE, fontsize=9)
    ax.set_xticks(lengths)
    ax.set_xlabel("Length of the input list")
    ax.set_ylabel("Items missing from [RESULT] (mean of 10 lists)")
    ax.set_ylim(0, 3.8)
    ax.set_title("Sorting: the minimum is always found, but long lists lose items",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sorting_items_lost.png"))
    plt.close(fig)


if __name__ == "__main__":
    addition_data_scaling()
    sorting_items_lost()
    print("Figures written to", OUT)
