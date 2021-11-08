"""
Bayesin Machine Learning in Python: A/B Testing
"""
from matplotlib.pyplot import plot, ylim, xscale, show
from numpy import empty, argmax, cumsum, arange, ones
from bayesian_bandit import Bandit


def run_experiment(p_1, p_2, p_3, n_0):
    """
    run experiment function
    """
    bandits = [Bandit(p_1), Bandit(p_2), Bandit(p_3)]
    data = empty(n_0)
    for i in range(n_0):
        j = argmax([b.sample() for b in bandits])
        x_var = bandits[j].pull()
        bandits[j].update(x_var)
        data[i] = x_var
    cumulative_average_ctr = cumsum(data) / (arange(n_0) + 1)
    plot(cumulative_average_ctr)
    plot(ones(n_0) * p_1)
    plot(ones(n_0) * p_2)
    plot(ones(n_0) * p_3)
    ylim((0, 1))
    xscale("log")
    show()


if __name__ == "__main__":
    run_experiment(0.2, 0.25, 0.3, 100000)
    show()
