"""
Hungry Machines: hunger regulation as a POMDP.

CONVENTION (fixed once, used everywhere):
    satiety index 0..4  ==  Starving, Very Hungry, Neutral, Full, Very Full
    Higher index = more full. "Getting hungrier" = index goes DOWN.

Hidden state   : true satiety (0-4)
Observation    : perceived satiety (0-4), noisy  -> stands in for interoception
Actions        : EAT / WAIT
"""

import random
import pomdp_py

LEVELS = ["Starving", "Very Hungry", "Neutral", "Full", "Very Full"]
N = len(LEVELS)


def clamp(i):
    return max(0, min(N - 1, i))


# ---------------------------------------------------------------- primitives

class HungerState(pomdp_py.State):
    def __init__(self, level):
        self.level = int(level)

    def __hash__(self):
        return hash(("S", self.level))

    def __eq__(self, o):
        return isinstance(o, HungerState) and o.level == self.level

    def __str__(self):
        return LEVELS[self.level]

    def __repr__(self):
        return f"HungerState({LEVELS[self.level]})"


class HungerObservation(pomdp_py.Observation):
    def __init__(self, level):
        self.level = int(level)

    def __hash__(self):
        return hash(("O", self.level))

    def __eq__(self, o):
        return isinstance(o, HungerObservation) and o.level == self.level

    def __str__(self):
        return f"feels:{LEVELS[self.level]}"

    def __repr__(self):
        return f"HungerObservation({LEVELS[self.level]})"


class HungerAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(("A", self.name))

    def __eq__(self, o):
        return isinstance(o, HungerAction) and o.name == self.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"HungerAction({self.name})"


EAT = HungerAction("eat")
WAIT = HungerAction("wait")
ACTIONS = [EAT, WAIT]


# ---------------------------------------------------------------- transition

class TransitionModel(pomdp_py.TransitionModel):
    """
    Metabolic drift is STOCHASTIC on purpose.

    Deterministic dynamics would let the agent dead-reckon its own hunger from
    its action history, the belief would stay a point mass, and observation
    noise would have literally no effect on behavior. That would make Step 6
    (low-noise vs high-noise comparison) produce a null result by construction.

    WAIT: satiety drifts down ~1 per step (sometimes 0, sometimes 2)
    EAT : satiety jumps up ~2       (sometimes 1, sometimes 3)
    """

    def __init__(self, drift_spread=0.15):
        # drift_spread = probability mass on each of the two off-nominal outcomes
        p = drift_spread
        self.wait_deltas = {-2: p, -1: 1 - 2 * p, 0: p}
        self.eat_deltas = {+1: p, +2: 1 - 2 * p, +3: p}

    def _deltas(self, action):
        return self.eat_deltas if action == EAT else self.wait_deltas

    def probability(self, next_state, state, action):
        total = 0.0
        for d, pr in self._deltas(action).items():
            if clamp(state.level + d) == next_state.level:
                total += pr  # mass folds onto the boundary when clamped
        return total

    def sample(self, state, action):
        deltas = self._deltas(action)
        d = random.choices(list(deltas.keys()), weights=list(deltas.values()))[0]
        return HungerState(clamp(state.level + d))

    def get_all_states(self):
        return [HungerState(i) for i in range(N)]


# --------------------------------------------------------------- observation

class ObservationModel(pomdp_py.ObservationModel):
    """
    Interoception. With prob `accuracy` the felt level matches the true level;
    otherwise it is off by one in either direction. Boundary mass folds inward.

    accuracy is THE experimental knob for Step 6.
    """

    def __init__(self, accuracy=0.8):
        self.accuracy = accuracy

    def _dist(self, true_level):
        off = (1.0 - self.accuracy) / 2.0
        dist = {}
        for d, pr in [(0, self.accuracy), (-1, off), (+1, off)]:
            lvl = clamp(true_level + d)
            dist[lvl] = dist.get(lvl, 0.0) + pr
        return dist

    def probability(self, observation, next_state, action):
        return self._dist(next_state.level).get(observation.level, 0.0)

    def sample(self, next_state, action):
        dist = self._dist(next_state.level)
        lvl = random.choices(list(dist.keys()), weights=list(dist.values()))[0]
        return HungerObservation(lvl)

    def get_all_observations(self):
        return [HungerObservation(i) for i in range(N)]


# ------------------------------------------------------------------- reward

class RewardModel(pomdp_py.RewardModel):
    """
    Discomfort is scored on the state you END UP in, plus a fixed cost to eat.
    (Scoring on next_state is what makes 'eating relieves hunger' actually pay.)

    The discomfort table is a parameter, not a constant, because Step 6 needs
    a symmetric-reward CONTROL arm to separate "overeats because the signal is
    unreliable" from "overeats because undereating is penalized harder".
    """

    # spec table: undershooting (-2) is twice as costly as overshooting (-1)
    ASYMMETRIC = {0: -2.0, 1: -1.0, 2: 0.0, 3: 0.0, 4: -1.0}
    # true mirror image: no directional incentive either way
    SYMMETRIC = {0: -2.0, 1: -1.0, 2: 0.0, 3: -1.0, 4: -2.0}

    EAT_COST = -0.5

    def __init__(self, discomfort=None, eat_cost=None):
        self.discomfort = dict(discomfort or RewardModel.ASYMMETRIC)
        self.eat_cost = RewardModel.EAT_COST if eat_cost is None else eat_cost

    def sample(self, state, action, next_state):
        r = self.discomfort[next_state.level]
        if action == EAT:
            r += self.eat_cost
        return r


# ------------------------------------------------------------------- policy

class PolicyModel(pomdp_py.RolloutPolicy):
    def sample(self, state):
        return random.choice(ACTIONS)

    def rollout(self, state, history=None):
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return ACTIONS


# ------------------------------------------------------------------ problem

class HungerProblem(pomdp_py.POMDP):
    def __init__(self, accuracy=0.8, init_state=None, drift_spread=0.15,
                 discomfort=None):
        init_state = init_state or HungerState(2)  # start Neutral
        init_belief = pomdp_py.Histogram(
            {HungerState(i): 1.0 / N for i in range(N)}  # uninformed prior
        )
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(),
            TransitionModel(drift_spread),
            ObservationModel(accuracy),
            RewardModel(discomfort),
        )
        env = pomdp_py.Environment(
            init_state, TransitionModel(drift_spread), RewardModel(discomfort)
        )
        super().__init__(agent, env, name="HungerProblem")


if __name__ == "__main__":
    # sanity checks
    tm, om = TransitionModel(), ObservationModel(0.8)

    for a in ACTIONS:
        for s in range(N):
            tot = sum(
                tm.probability(HungerState(j), HungerState(s), a) for j in range(N)
            )
            assert abs(tot - 1.0) < 1e-9, (a, s, tot)
    print("transition rows sum to 1  ok")

    for s in range(N):
        tot = sum(
            om.probability(HungerObservation(j), HungerState(s), WAIT)
            for j in range(N)
        )
        assert abs(tot - 1.0) < 1e-9, (s, tot)
    print("observation rows sum to 1 ok")

    p = HungerProblem()
    print("problem builds ok        ok")
    print("start:", p.env.state)
