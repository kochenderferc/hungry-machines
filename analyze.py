"""
Step 6 statistics. Run after sweep.py has produced sweep_results.csv.

Design: 2 (reward regime) x 4 (interoceptive accuracy), 10 seeds per cell,
between-seeds. Each seed is an independent 100-step episode, so cells are
independent samples and unpaired tests are appropriate.

Reports, for each outcome:
  1. within-regime effect of noise (accuracy 0.95 vs 0.45), Welch's t
  2. the regime x noise INTERACTION, which is the headline result
  3. Hedges' g effect sizes, because the p-values here outrun the effect sizes
"""

import numpy as np
from scipy import stats

from sweep import load_rows, ACCURACIES, REGIMES

OUTCOMES = ["eat_rate", "gap_sd", "mean_trigger",
            "pct_very_full", "pct_starving", "mean_reward"]

HI, LO = 0.95, 0.45  # cleanest signal vs noisiest signal


def cell(rows, regime, acc, metric):
    return np.array([r[metric] for r in rows
                     if r["regime"] == regime and r["accuracy"] == acc])


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    correction = 1 - (3 / (4 * (na + nb) - 9))  # small-sample bias correction
    return d * correction


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def main(path="sweep_results.csv"):
    rows = load_rows(path)

    print("=" * 78)
    print("1. EFFECT OF NOISE WITHIN EACH REWARD REGIME  "
          f"(accuracy {HI} vs {LO}, Welch's t)")
    print("=" * 78)
    for regime in REGIMES:
        print(f"\n  {regime} rewards")
        for m in OUTCOMES:
            a, b = cell(rows, regime, HI, m), cell(rows, regime, LO, m)
            t, p = stats.ttest_ind(a, b, equal_var=False)
            g = hedges_g(a, b)
            print(f"    {m:<16}{a.mean():+.3f} -> {b.mean():+.3f}   "
                  f"t={t:+6.2f}  p={p:.4f} {stars(p):<3}  g={g:+.2f}")

    print("\n" + "=" * 78)
    print("2. REGIME x NOISE INTERACTION")
    print("   H0: noise moves both regimes the same way.")
    print("   Tests the difference-of-differences across independent cells.")
    print("=" * 78 + "\n")
    for m in OUTCOMES:
        d_asym = (cell(rows, "asymmetric", LO, m)
                  - cell(rows, "asymmetric", HI, m).mean())
        d_sym = (cell(rows, "symmetric", LO, m)
                 - cell(rows, "symmetric", HI, m).mean())
        t, p = stats.ttest_ind(d_asym, d_sym, equal_var=False)
        print(f"  {m:<16}delta_asym={d_asym.mean():+.3f}  "
              f"delta_sym={d_sym.mean():+.3f}   "
              f"t={t:+6.2f}  p={p:.4f} {stars(p)}")

    print("\n" + "=" * 78)
    print("3. MAIN EFFECT OF REWARD REGIME (pooled over accuracy)")
    print("=" * 78 + "\n")
    for m in OUTCOMES:
        a = np.array([r[m] for r in rows if r["regime"] == "asymmetric"])
        b = np.array([r[m] for r in rows if r["regime"] == "symmetric"])
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"  {m:<16}asym={a.mean():+.3f}  sym={b.mean():+.3f}   "
              f"t={t:+6.2f}  p={p:.4f} {stars(p):<3}  g={hedges_g(a, b):+.2f}")

    print("\n" + "=" * 78)
    print("4. MONOTONICITY ACROSS ALL FOUR ACCURACY LEVELS (Spearman)")
    print("   A real dose-response should be monotonic, not just endpoint-different.")
    print("=" * 78 + "\n")
    for regime in REGIMES:
        print(f"  {regime}")
        for m in OUTCOMES:
            xs, ys = [], []
            for acc in ACCURACIES:
                v = cell(rows, regime, acc, m)
                xs += [acc] * len(v)
                ys += list(v)
            rho, p = stats.spearmanr(xs, ys)
            print(f"    {m:<16}rho={rho:+.3f}  p={p:.4f} {stars(p)}")
        print()


if __name__ == "__main__":
    main()
