import numpy as np


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def step(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class EpsilonGreedyPolicy:
    def __init__(self, greedy_policy, random_policy, exploration_schedule):
        self.greedy_policy = greedy_policy
        self.random_policy = random_policy
        self.exploration_schedule = exploration_schedule

    def __call__(self, obs) -> int:
        u = np.random.rand()
        if u < self.exploration_schedule.current:
            action = self.random_policy(obs)
        else:
            action = self.greedy_policy(obs)

        return action

    @property
    def epsilon(self):
        return self.exploration_schedule.current
