"""
Bayesin Machine Learning in Python: A/B Testing
"""
from matplotlib.pyplot import show, plot, ylim
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    """
    bandit class
    """

    def __init__(self, p):
        self.p_var = p
        self.p_estimate = 5.0
        self.n_var = 1.0

    def pull(self):
        """
        pull method
        """
        return np.random.random() < self.p_var

    def update(self, x_var):
        """
        update method
        """
        self.n_var += 1.0
        self.p_estimate = ((self.n_var - 1) * self.p_estimate + x_var) / self.n_var


def experiment():
    """
    experiment function
    """
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = np.argmax([b.p_estimate for b in bandits])
        x_var = bandits[j].pull()
        rewards[i] = x_var
        bandits[j].update(x_var)
    for b_var in bandits:
        print("mean estimate:", b_var.p_estimate)
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.n_var for b in bandits])
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    ylim([0, 1])
    plot(win_rates)
    plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))


if __name__ == "__main__":
    experiment()
    show()
