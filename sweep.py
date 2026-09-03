"""
Step 6: does interoceptive noise change eating behavior?

2 x 4 factorial, multiple seeds per cell:

    reward regime  : ASYMMETRIC (spec)  vs  SYMMETRIC (control)
    accuracy       : 0.95, 0.80, 0.60, 0.45

The control arm exists to answer the confound. In the spec's reward table
undershooting (-2 Starving) costs twice what overshooting (-1 Very Full) does,
so a rational agent overeats on purpose regardless of signal quality. The
symmetric table removes that directional incentive. Any noise effect that
survives symmetrization is attributable to signal reliability; any effect that
disappears was loss asymmetry all along.
"""

import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hunger_pomdp import RewardModel, LEVELS, N
from simulate import run_episode, summarize, BURN_IN

ACCURACIES = [0.95, 0.80, 0.60, 0.45]
REGIMES = {
    "asymmetric": RewardModel.ASYMMETRIC,
    "symmetric": RewardModel.SYMMETRIC,
}
N_SEEDS = 30
STEPS = 100
NUM_SIMS = 400

METRICS = ["eat_rate", "gap_sd", "mean_gap", "mean_trigger",
           "pct_very_full", "pct_starving", "pct_comfortable",
           "mean_reward", "mean_entropy"]


def run_sweep(path="sweep_results.csv", only_regime=None, append=False):
    rows = []
    regimes = ({only_regime: REGIMES[only_regime]} if only_regime else REGIMES)
    total = len(regimes) * len(ACCURACIES) * N_SEEDS
    done = 0

    for regime, table in regimes.items():
        for acc in ACCURACIES:
            for seed in range(N_SEEDS):
                log = run_episode(accuracy=acc, steps=STEPS, seed=seed,
                                  num_sims=NUM_SIMS, discomfort=table)
                s = summarize(log, f"{regime}/{acc}")
                row = {"regime": regime, "accuracy": acc, "seed": seed}
                row.update({m: float(s[m]) for m in METRICS})
                row["trigger_dist"] = json.dumps(s["trigger_dist"])
                rows.append(row)
                done += 1
                print(f"[{done:>3}/{total}] {regime:<10} acc={acc:.2f} "
                      f"seed={seed}  eat_rate={s['eat_rate']:.3f}",
                      flush=True)

    import os
    mode = "a" if (append and os.path.exists(path)) else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    return rows


def load_rows(path="sweep_results.csv"):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["accuracy"] = float(r["accuracy"])
            r["seed"] = int(r["seed"])
            for m in METRICS:
                r[m] = float(r[m])
            out.append(r)
    return out


def aggregate(rows):
    """mean and standard error per (regime, accuracy) cell."""
    agg = {}
    for regime in REGIMES:
        for acc in ACCURACIES:
            cell = [r for r in rows
                    if r["regime"] == regime and r["accuracy"] == acc]
            agg[(regime, acc)] = {
                m: (float(np.mean([c[m] for c in cell])),
                    float(np.std([c[m] for c in cell], ddof=1) / np.sqrt(len(cell))))
                for m in METRICS
            }
    return agg


def plot_sweep(agg, path="noise_sweep.png"):
    panels = [
        ("eat_rate", "Eating frequency", "fraction of steps"),
        ("gap_sd", "Erraticness of meal timing", "SD of inter-meal gap"),
        ("mean_trigger", "Satiety level when eating", "0=Starving .. 4=Very Full"),
        ("pct_very_full", "Time spent Very Full", "fraction of steps"),
        ("pct_starving", "Time spent Starving", "fraction of steps"),
        ("mean_reward", "Regulation quality", "mean reward/step"),
    ]
    colors = {"asymmetric": "tab:red", "symmetric": "tab:blue"}
    markers = {"asymmetric": "o", "symmetric": "s"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    x = np.array(ACCURACIES)

    for ax, (metric, title, ylab) in zip(axes.ravel(), panels):
        for regime in REGIMES:
            mu = np.array([agg[(regime, a)][metric][0] for a in ACCURACIES])
            se = np.array([agg[(regime, a)][metric][1] for a in ACCURACIES])
            ax.errorbar(x, mu, yerr=se, marker=markers[regime], capsize=3,
                        lw=1.8, color=colors[regime],
                        label=f"{regime} reward")
            ax.fill_between(x, mu - se, mu + se, color=colors[regime], alpha=0.12)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("interoceptive accuracy")
        ax.set_ylabel(ylab, fontsize=8)
        ax.invert_xaxis()          # left = reliable signal, right = noisy
        ax.grid(alpha=0.25)

    axes[0][0].legend(fontsize=8)
    fig.suptitle(
        "Step 6: effect of interoceptive noise, with symmetric-reward control "
        f"({N_SEEDS} seeds/cell, {STEPS} steps, burn-in {BURN_IN} excluded)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_table(agg):
    hdr = f"{'regime':<11}{'acc':>6}" + "".join(f"{m:>16}" for m in
          ["eat_rate", "gap_sd", "mean_trigger", "pct_very_full", "mean_reward"])
    print("\n" + hdr)
    print("-" * len(hdr))
    for regime in REGIMES:
        for acc in ACCURACIES:
            c = agg[(regime, acc)]
            line = f"{regime:<11}{acc:>6.2f}"
            for m in ["eat_rate", "gap_sd", "mean_trigger",
                      "pct_very_full", "mean_reward"]:
                line += f"{c[m][0]:>10.3f}±{c[m][1]:.3f}"
            print(line)
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in REGIMES:
        run_sweep(only_regime=sys.argv[1], append=(sys.argv[1] != "asymmetric"))
        print("chunk done:", sys.argv[1])
        sys.exit(0)
    rows = load_rows()
    agg = aggregate(rows)
    print_table(agg)
    plot_sweep(agg)
    print("wrote sweep_results.csv and noise_sweep.png")
