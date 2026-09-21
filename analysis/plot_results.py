#!/usr/bin/env python3
"""Replot the published numerical figure data on CPU, without experiment logs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"RandOpt": "#63758a", "Modular Norm RandOpt": "#bd3c44", "MN RandOpt": "#bd3c44"}


def save(fig, output, name):
    fig.tight_layout()
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def curve(ax, x, y, sd, label, *, bounds=None):
    x, y, sd = (np.asarray(value, dtype=float) for value in (x, y, sd))
    lower, upper = y - sd, y + sd
    if bounds:
        lower, upper = np.clip(lower, *bounds), np.clip(upper, *bounds)
    color = COLORS.get(label, "#795ca4")
    ax.plot(x, y, label=label, color=color, linewidth=1.8)
    ax.fill_between(x, lower, upper, color=color, alpha=.18, linewidth=0)
    ax.grid(alpha=.2)


def figure1(data, output):
    scaling = pd.read_csv(data / "figure1_population_scaling.csv")
    transfer = pd.read_csv(data / "figure1_transfer_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].axis("off")
    for i, text in enumerate(("Pretrained model", "Architecture-aware perturbations",
                               "Select top-K by training reward", "Ensemble by answer voting")):
        y = .92 - i * .26
        axes[0].text(.5, y, text, ha="center", va="center", fontsize=10,
                     bbox=dict(boxstyle="round,pad=.45", fc="#f5f1ed", ec="#777"))
        if i < 3:
            axes[0].annotate("", xy=(.5, y-.18), xytext=(.5, y-.07), arrowprops=dict(arrowstyle="->"))
    for method, group in scaling[scaling.task == "countdown"].groupby("method", sort=False):
        group = group.sort_values("population_size")
        curve(axes[1], group.population_size, group.accuracy_mean, group.accuracy_std, method)
    axes[1].set(xlabel="Candidates N", ylabel="Accuracy (%)", title="Countdown, K=25")
    axes[1].legend(fontsize=8)
    table = transfer.pivot(index="model", columns="task", values="mn_randopt_minus_randopt_pt")
    limit = max(abs(table.min().min()), abs(table.max().max()))
    im = axes[2].imshow(table, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[2].set_xticks(range(len(table.columns)), table.columns, rotation=65, ha="right", fontsize=7)
    axes[2].set_yticks(range(len(table.index)), table.index, fontsize=7)
    axes[2].set_title("MN − RandOpt (pp)")
    fig.colorbar(im, ax=axes[2], shrink=.8)
    save(fig, output, "figure1")


def figure2(data, output):
    frame = pd.read_csv(data / "figure2_population_scaling_data.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for row, k in enumerate((10, 25)):
        for col, task in enumerate(("countdown", "gsm8k")):
            ax = axes[row, col]
            subset = frame[(frame.task == task) & (frame.k == k)]
            for method, group in subset.groupby("method", sort=False):
                group = group.sort_values("population_size")
                curve(ax, group.population_size, group.accuracy_mean_percent,
                      group.accuracy_sample_sd_percent, method)
            ax.set(title=f"{task.upper() if task == 'gsm8k' else 'Countdown'}, K={k}",
                   xlabel="Candidates N", ylabel="Accuracy (%)")
            ax.legend(fontsize=8)
    save(fig, output, "figure2")


def figure3(data, output):
    frame = pd.read_csv(data / "figure3_candidate_survival.csv")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for ax, (task, subset) in zip(axes.flat, frame.groupby("task", sort=False)):
        for method, group in subset.groupby("method", sort=False):
            curve(ax, group.reward_gain_threshold_pp, group.tail_probability_mean_pct,
                  group.tail_probability_sample_sd_pct, method, bounds=(0, 100))
        ax.set(title=subset.task_label.iloc[0], xlabel="Selection-reward gain (pp)", ylabel="Candidates (%)")
        ax.set_ylim(0, 100)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=7)
    save(fig, output, "figure3")


def figure4(data, output):
    density = pd.read_csv(data / "figure4_useful_candidate_density.csv")
    ratio = pd.read_csv(data / "figure4_required_population_ratio.csv")
    cutoffs = pd.read_csv(data / "figure4_top10_cutoff_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for method, group in density.groupby("method", sort=False):
        curve(axes[0], group.threshold_above_base_percentage_points, group.density_mean_percent,
              group.density_sample_sd_percent, method, bounds=(.01, 100))
    for row in cutoffs.to_dict("records"):
        axes[0].axvspan(row["mean"]-row["sample_sd"], row["mean"]+row["sample_sd"],
                        color=COLORS.get(row["method"]), alpha=.12)
    axes[0].set(yscale="log", ylabel="Candidates above threshold (%)")
    axes[0].legend(fontsize=8)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna(subset=["tail_implied_candidate_reduction_ratio", "ratio_sample_sd"])
    curve(axes[1], ratio.threshold_above_base_percentage_points,
          ratio.tail_implied_candidate_reduction_ratio, ratio.ratio_sample_sd, "MN RandOpt", bounds=(.01, 100))
    axes[1].axhline(1, color="#777", linewidth=.8)
    axes[1].axhline(12, color="#555", linestyle="--", label="Observed ≥12×")
    axes[1].set(yscale="log", xlabel="Selection-reward gain threshold (pp)", ylabel="Required-population ratio")
    axes[1].legend(fontsize=8)
    save(fig, output, "figure4")


def figure5(data, output):
    support = pd.read_csv(data / "figure5_support_aggregate.csv")
    decomposition = pd.read_csv(data / "figure5_exact_decomposition_aggregate.csv").iloc[0]
    seeds = pd.read_csv(data / "figure5_exact_decomposition_per_seed.csv")
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6))
    x = np.arange(len(support))
    for delta, method, prefix in ((-.19, "RandOpt", "randopt"), (.19, "MN RandOpt", "modular_norm_randopt")):
        axes[0].bar(x+delta, support[f"{prefix}_mean"]*100, .36,
                    yerr=support[f"{prefix}_sample_sd"]*100, capsize=3, color=COLORS[method], label=method)
    axes[0].set_xticks(x, support.support_bin)
    axes[0].set(xlabel="Number of correct experts", ylabel="Evaluation problems (%)")
    axes[0].legend()
    columns = ("support_redistribution_contribution", "conditional_vote_contribution", "ensemble_accuracy_delta")
    labels = ("Support redistribution", "Conditional voting", "Total gain")
    for i, key in enumerate(columns):
        axes[1].barh(i, 100*decomposition[f"{key}_mean"], xerr=100*decomposition[f"{key}_sample_sd"], capsize=3)
        axes[1].scatter(seeds[key]*100, np.full(len(seeds), i), color="black", s=12, zorder=3)
    axes[1].set_yticks(range(3), labels)
    axes[1].set_xlabel("MN − RandOpt (pp)")
    axes[1].axvline(0, color="#555", linewidth=.7)
    save(fig, output, "figure5")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "results")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/figures")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for draw in (figure1, figure2, figure3, figure4, figure5):
        draw(args.data, args.output)
    print(f"Created Figures 1–5 from published CSV files in {args.output}")


if __name__ == "__main__":
    main()
