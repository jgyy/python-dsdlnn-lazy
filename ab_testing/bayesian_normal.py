"""
https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
"""
from numpy import sqrt, linspace, empty, argmax, ones, cumsum, arange
from numpy.random import randn
from matplotlib.pyplot import plot as pplot, show, title, legend, figure
from scipy.stats import norm

NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]


class Bandit:
    """
    bandit class
    """

    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.m_data = 0
        self.lambda_ = 1
        self.tau = 1
        self.n_data = 0

    def pull(self):
        """
        pull method
        """
        return randn() / sqrt(self.tau) + self.true_mean

    def sample(self):
        """
        sample method
        """
        return randn() / sqrt(self.lambda_) + self.m_data

    def update(self, x_data):
        """
        update method
        """
        self.m_data = (self.tau * x_data + self.lambda_ * self.m_data) / (
            self.tau + self.lambda_
        )
        self.lambda_ += self.tau
        self.n_data += 1


def plot(bandits, trial):
    """
    plot function
    """
    x_data = linspace(-3, 6, 200)
    figure()
    for b_data in bandits:
        y_data = norm.pdf(x_data, b_data.m_data, sqrt(1.0 / b_data.lambda_))
        pplot(
            x_data,
            y_data,
            label=f"real mean: {b_data.true_mean:.4f}, num plays: {b_data.n_data}",
        )
    title(f"Bandit distributions after {trial} trials")
    legend()


def run_experiment():
    """
    run experiment function
    """
    bandits = [Bandit(m) for m in BANDIT_MEANS]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = empty(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = argmax([b.sample() for b in bandits])
        if i in sample_points:
            plot(bandits, i)
        x_data = bandits[j].pull()
        bandits[j].update(x_data)
        rewards[i] = x_data
    cumulative_average = cumsum(rewards) / (arange(NUM_TRIALS) + 1)
    pplot(cumulative_average)
    for m_data in BANDIT_MEANS:
        pplot(ones(NUM_TRIALS) * m_data)
    return cumulative_average


if __name__ == "__main__":
    run_experiment()
    show()
