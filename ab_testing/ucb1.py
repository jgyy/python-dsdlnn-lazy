"""
https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
"""
from numpy import sqrt, log, empty, argmax, cumsum, arange, ones, max as nmax
from numpy.random import random
from matplotlib.pyplot import show, plot, xscale, figure

NUM_TRIALS = 100000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    """
    bandit class
    """

    def __init__(self, p):
        self.p_var = p
        self.p_estimate = 0.0
        self.n_var = 0.0

    def pull(self):
        """
        draw a 1 with probability p
        """
        return random() < self.p_var

    def update(self, x_var):
        """
        update method
        """
        self.n_var += 1.0
        self.p_estimate = ((self.n_var - 1) * self.p_estimate + x_var) / self.n_var


def ucb(mean, n_var, nj_var):
    """
    ucb function
    """
    return mean + sqrt(2 * log(n_var) / nj_var)


def run_experiment():
    """
    run experiment function
    """
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = empty(NUM_TRIALS)
    total_plays = 0
    for j in bandits:
        x_var = j.pull()
        total_plays += 1
        j.update(x_var)
    for i in range(NUM_TRIALS):
        j = argmax([ucb(b.p_estimate, total_plays, b.n_var) for b in bandits])
        x_var = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x_var)
        rewards[i] = x_var
    cumulative_average = cumsum(rewards) / (arange(NUM_TRIALS) + 1)
    figure()
    plot(cumulative_average)
    plot(ones(NUM_TRIALS) * nmax(BANDIT_PROBABILITIES))
    xscale("log")
    figure()
    plot(cumulative_average)
    plot(ones(NUM_TRIALS) * nmax(BANDIT_PROBABILITIES))
    for b_var in bandits:
        print(b_var.p_estimate)
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.n_var for b in bandits])
    return cumulative_average


if __name__ == "__main__":
    run_experiment()
    show()
