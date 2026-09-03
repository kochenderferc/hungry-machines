"""
Steps 4-5: run the online planner over an extended horizon, record the
trajectory, and plot true hunger with eating decisions marked.

Because the agent starts from a UNIFORM prior it does not initially know how
hungry it is, so the first few timesteps are inference startup rather than
steady-state regulation. BURN_IN marks that window; summary statistics are
computed after it, and the plots shade it so it stays visible rather than
silently discarded.
"""

import math
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pomdp_py

from hunger_pomdp import (
    HungerProblem, HungerState, LEVELS, N, EAT, WAIT,
)

BURN_IN = 10


def belief_stats(belief):
    """Expected satiety and Shannon entropy (nats) of the agent's belief."""
    ps = np.array([belief[HungerState(i)] for i in range(N)])
    ps = ps / ps.sum()
    mean = float((ps * np.arange(N)).sum())
    ent = float(-(ps * np.log(ps + 1e-12)).sum())
    return mean, ent


def run_episode(accuracy=0.8, drift_spread=0.15, steps=100, seed=0,
                num_sims=800, max_depth=15, discount=0.95, discomfort=None):
    """One full simulated trajectory. Returns a dict of per-timestep arrays."""
    random.seed(seed)
    np.random.seed(seed)

    prob = HungerProblem(accuracy=accuracy, drift_spread=drift_spread,
                         discomfort=discomfort)
    planner = pomdp_py.POUCT(
        max_depth=max_depth,
        discount_factor=discount,
        num_sims=num_sims,
        exploration_const=8,
        rollout_policy=prob.agent.policy_model,
    )

    log = {k: [] for k in
           ["true", "true_pre", "perceived", "ate", "reward",
            "bel_mean", "bel_entropy"]}

    for _ in range(steps):
        # state the agent was actually in when it made this decision
        pre_level = prob.env.state.level

        action = planner.plan(prob.agent)

        # environment advances and pays out reward
        reward = prob.env.state_transition(action, execute=True)

        # the agent only ever sees this noisy reading -- never env.state
        obs = prob.agent.observation_model.sample(prob.env.state, action)

        prob.agent.update_history(action, obs)
        planner.update(prob.agent, action, obs)

        # exact Bayes update (tractable: only 5 states)
        new_belief = pomdp_py.update_histogram_belief(
            prob.agent.cur_belief, action, obs,
            prob.agent.observation_model, prob.agent.transition_model,
        )
        prob.agent.set_belief(new_belief)

        mean, ent = belief_stats(new_belief)
        log["true"].append(prob.env.state.level)
        log["true_pre"].append(pre_level)
        log["perceived"].append(obs.level)
        log["ate"].append(action == EAT)
        log["reward"].append(reward)
        log["bel_mean"].append(mean)
        log["bel_entropy"].append(ent)

    return {k: np.array(v) for k, v in log.items()}


def summarize(log, label, burn_in=BURN_IN):
    """Steady-state behavioral measures, burn-in excluded."""
    true = log["true"][burn_in:]
    ate = log["ate"][burn_in:]
    rew = log["reward"][burn_in:]

    gaps = np.diff(np.flatnonzero(ate)) if ate.sum() > 1 else np.array([np.nan])

    # at what TRUE satiety level does the agent pull the trigger?
    pre = log["true_pre"][burn_in:]
    trigger = pre[ate]
    trigger_dist = {LEVELS[i]: float((trigger == i).mean()) if len(trigger) else 0.0
                    for i in range(N)}

    return {
        "label": label,
        "eat_rate": ate.mean(),
        "mean_gap": np.nanmean(gaps),
        "gap_sd": np.nanstd(gaps),          # erraticness of meal timing
        "mean_trigger": trigger.mean() if len(trigger) else np.nan,
        "trigger_dist": trigger_dist,
        "pct_starving": (true == 0).mean(),
        "pct_very_full": (true == N - 1).mean(),
        "pct_comfortable": np.isin(true, [2, 3]).mean(),
        "mean_reward": rew.mean(),
        "mean_entropy": log["bel_entropy"][burn_in:].mean(),
    }


def plot_trajectory(log, title, path, burn_in=BURN_IN):
    t = np.arange(len(log["true"]))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.axhspan(1.5, 3.5, color="tab:green", alpha=0.07, zorder=0)
    ax1.axvspan(0, burn_in, color="gray", alpha=0.12, zorder=0)
    ax1.text(burn_in / 2, 4.35, "burn-in", ha="center", fontsize=8, color="dimgray")

    ax1.plot(t, log["true"], color="tab:blue", lw=1.8,
             label="true satiety (hidden)", zorder=3)
    ax1.plot(t, log["bel_mean"], color="tab:orange", lw=1.2, ls="--",
             alpha=0.9, label="agent's expected satiety", zorder=2)
    ax1.scatter(t[log["perceived"] != log["true"]],
                log["perceived"][log["perceived"] != log["true"]],
                s=14, color="tab:red", alpha=0.55, marker="x",
                label="misperceived signal", zorder=4)

    # markers sit at the satiety level the agent was in WHEN IT DECIDED,
    # not the level it landed on afterwards
    eat_t = t[log["ate"]]
    ax1.scatter(eat_t, log["true_pre"][log["ate"]], s=70, color="black",
                marker="v", zorder=5, label="chose to eat (level at decision)")

    ax1.set_yticks(range(N))
    ax1.set_yticklabels(LEVELS)
    ax1.set_ylim(-0.5, 4.6)
    ax1.set_ylabel("satiety level")
    ax1.set_title(title)
    ax1.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax1.grid(alpha=0.2)

    ax2.plot(t, log["bel_entropy"], color="tab:purple", lw=1.3)
    ax2.axvspan(0, burn_in, color="gray", alpha=0.12)
    ax2.set_ylabel("belief\nentropy (nats)", fontsize=9)
    ax2.set_xlabel("time step")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    log = run_episode(accuracy=0.8, steps=100, seed=7)
    plot_trajectory(
        log,
        "Hunger regulation under uncertain interoception (accuracy = 0.80)",
        "hunger_trajectory.png",
    )
    s = summarize(log, "accuracy 0.80")
    for k, v in s.items():
        print(f"{k:>16}: {v if isinstance(v, str) else round(float(v), 3)}")
