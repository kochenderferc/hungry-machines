# Hungry Machines

Simulating interoceptive uncertainty in eating behavior with a POMDP agent.

Hunger is treated as a **hidden state**; the agent never observes it directly and
must act on a noisy internal signal standing in for interoception. An online
POMDP planner chooses to eat or wait from the agent's current belief at each
timestep.

This is a computational model. It makes **no clinical claims** and uses no human
subject data.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
python hunger_pomdp.py        # model sanity checks (probability rows sum to 1)
python simulate.py            # single 100-step episode -> hunger_trajectory.png
python sweep.py asymmetric    # experiment arm 1  (~12 min, rewrites CSV)
python sweep.py symmetric     # experiment arm 2  (~12 min, appends)
python sweep.py               # aggregate + noise_sweep.png
python analyze.py             # statistics
```

Runs are seed-reproducible: identical seeds produce byte-identical trajectories.

## Model

Satiety index `0..4` = Starving, Very Hungry, Neutral, Full, Very Full.
Higher index = more full, so "getting hungrier" means the index goes **down**.
This convention is fixed once and used everywhere; the original project spec
described transitions in terms of *hunger* rising while ordering the levels by
*fullness*, which inverts the sign if left implicit.

| component | definition |
|---|---|
| hidden state | true satiety, 5 levels |
| observation | perceived satiety, 5 levels, noisy (interoception) |
| actions | `EAT` / `WAIT` |
| transition | `WAIT`: satiety −1 (±1 metabolic drift). `EAT`: +2 (±1) |
| observation model | correct with prob `accuracy`, else off-by-one either way |
| reward | discomfort scored on the state reached, −0.5 to eat |

### Two deliberate departures from the original spec

**1. Transitions are stochastic, not deterministic.**
With deterministic dynamics the belief update
`b'(s') ∝ O(o|s') · Σ T(s'|s,a) b(s)` collapses: if `T` is a delta and the prior
is a point mass, the posterior stays a point mass and the observation term
normalizes away. The agent could dead-reckon its hunger from its own action
history and interoceptive noise would have **no** behavioral effect, making the
noise experiment a null result by construction. Measured mean belief entropy:
0.031 nats deterministic vs 0.428 nats with drift.

**2. Exact belief updates (POUCT) rather than particle filtering (POMCP).**
Same planner; POMCP is the particle-belief variant. With only 5 states the exact
Bayes update is closed-form, which removes particle depletion as a competing
explanation for any behavioral difference. Swapping in POMCP is a one-line change.

Note that POUCT/POMCP are **online anytime planners**. They replan from the
current belief every timestep and never produce a stored policy — nothing is
"learned" in the reinforcement-learning sense.

## Experiment

2 (reward regime) × 4 (interoceptive accuracy: 0.95 / 0.80 / 0.60 / 0.45),
30 seeds per cell, 100 steps each, first 10 steps discarded as inference burn-in
(the agent starts from a uniform prior and does not initially know how hungry it is).

The **symmetric-reward control arm** is the methodological core. The spec's
reward table penalizes Starving (−2) twice as hard as Very Full (−1), so a
rational agent overshoots on purpose regardless of signal quality. The symmetric
table (−2/−1/0/−1/−2) removes that directional incentive.

## Findings

**Overeating is driven by loss asymmetry, not by interoceptive noise.** Pooled
across accuracy levels, time spent Very Full is 0.282 under asymmetric rewards
versus 0.055 under symmetric (Hedges' g = 7.44), and satiety-at-eating differs
by g = 8.68. These are the largest effects in the study by an order of magnitude.

**Noise reliably changes eating frequency only when the asymmetry is removed.**
Under symmetric rewards, noisier agents eat less (0.342 -> 0.324, p = .0003,
g = 1.00; Spearman rho = +0.355 across all four accuracy levels). Under
asymmetric rewards there is no reliable effect (p = .10, rho = -0.153, ns).
The regime x noise interaction is significant (p < .0001).

**What noise does robustly, in both regimes:** it degrades regulation quality
(mean reward, rho = +0.549 and +0.589, both p < .0001, g = 1.60 and 2.03) and
increases time spent Starving (rho = -0.413 and -0.613, both p < .0001,
g = 1.26 and 2.10).

**Under symmetric costs, noise pushes the agent toward both extremes at once** —
Starving rises (0.062 -> 0.132) *and* Very Full rises (0.041 -> 0.064,
p = .0005). This is a loss-of-control signature rather than a directional bias.

**Noise also shifts the policy, not just outcomes.** Under symmetric rewards,
noisier agents eat at systematically lower satiety (rho = +0.461, g = 1.50).

The intuitive hypothesis -- unreliable interoception causes overeating -- is
**not supported**. The direction of the interoception-eating relationship is
underdetermined without specifying the costs the agent assigns to error.

## Limitations

- 5 levels is coarse relative to a ±1/+2 step size; the agent spends much of its
  time pinned against the Very Full ceiling.
- Effect sizes on eating frequency are small in absolute terms (~1-2 pp) even
  where p-values are small.
- Inter-meal-gap SD (`gap_sd`) was unstable across sample sizes, changing
  direction and significance between a 10-seed pilot and the 30-seed run.
  It is reported as exploratory and no claim rests on it.
- The reward table is hand-specified, not fitted, and it turned out to drive
  the largest effects in the study. Results are conditional on that choice.
- Discretized state space, no human subject data.
- Metabolic drift parameters are chosen for the modeling reason above, not
  fitted to physiological data.

## Files

| file | purpose |
|---|---|
| `hunger_pomdp.py` | state/observation/action spaces, transition, observation, reward |
| `simulate.py` | single-episode planner loop, trajectory plot |
| `sweep.py` | 2×4 factorial experiment, aggregation, comparison figure |
| `analyze.py` | t-tests, interaction, effect sizes, monotonicity |
| `sweep_results.csv` | raw per-episode results |
