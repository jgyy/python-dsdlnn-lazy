"""
Bayesin Machine Learning in Python: A/B Testing
"""
from matplotlib.pyplot import plot as pplot, show, title, legend, figure
from numpy import linspace, zeros, argmax
from numpy.random import random, beta
from scipy.stats import beta as sbeta

NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    """
    bandit class
    """

    def __init__(self, p_var):
        self.p_var = p_var
        self.a_var = 1
        self.b_var = 1
        self.n_var = 0

    def pull(self):
        """
        pull method
        """
        return random() < self.p_var

    def sample(self):
        """
        sample method
        """
        return beta(self.a_var, self.b_var)

    def update(self, x_var):
        """
        update method
        """
        self.a_var += x_var
        self.b_var += 1 - x_var
        self.n_var += 1


def plot(bandits, trial):
    """
    plot function
    """
    x_data = linspace(0, 1, 200)
    figure()
    for b_data in bandits:
        y_data = sbeta.pdf(x_data, b_data.a_var, b_data.b_var)
        pplot(
            x_data,
            y_data,
            label=f"real p: {b_data.p_var:.4f}, win rate = {b_data.a_var - 1}/{b_data.n_var}",
        )
    title(f"Bandit distributions after {trial} trials")
    legend()


def experiment():
    """
    experiment function
    """
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = argmax([b.sample() for b in bandits])
        if i in sample_points:
            plot(bandits, i)
        x_data = bandits[j].pull()
        rewards[i] = x_data
        bandits[j].update(x_data)
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.n_var for b in bandits])


if __name__ == "__main__":
    experiment()
    ban = Bandit(0.1)
    print(ban.pull())
    print(ban.sample())
    assert isinstance(ban.pull(), bool)
    assert 0.1 <= ban.sample() <= 1
    show()
