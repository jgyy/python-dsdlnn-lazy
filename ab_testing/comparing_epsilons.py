"""
https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
"""
from numpy import argmax, array, empty, cumsum, arange, ones
from numpy.random import random, choice, randn
from matplotlib.pyplot import plot, show, figure, xscale, legend


class BanditArm:
    """
    bandit arm class
    """

    def __init__(self, m_var):
        self.m_var = m_var
        self.m_estimate = 0
        self.n_var = 0

    def pull(self):
        """
        pull method
        """
        return randn() + self.m_var

    def update(self, x_var):
        """
        update method
        """
        self.n_var += 1
        self.m_estimate = (
            1 - 1.0 / self.n_var
        ) * self.m_estimate + 1.0 / self.n_var * x_var


def run_experiment(m_1, m_2, m_3, eps, n_0):
    """
    run experiment function
    """
    bandits = [BanditArm(m_1), BanditArm(m_2), BanditArm(m_3)]
    true_best = argmax(array([m_1, m_2, m_3]))
    count_suboptimal = 0
    data = empty(n_0)
    for i in range(n_0):
        p_var = random()
        if p_var < eps:
            j = choice(len(bandits))
        else:
            j = argmax([b.m_estimate for b in bandits])
        x_var = bandits[j].pull()
        bandits[j].update(x_var)
        if j != true_best:
            count_suboptimal += 1
        data[i] = x_var
    cumulative_average = cumsum(data) / (arange(n_0) + 1)
    figure()
    plot(cumulative_average)
    plot(ones(n_0) * m_1)
    plot(ones(n_0) * m_2)
    plot(ones(n_0) * m_3)
    xscale("log")
    for b_var in bandits:
        print(b_var.m_estimate)
    print("percent suboptimal for epsilon = %s:" % eps, float(count_suboptimal) / n_0)
    return cumulative_average


if __name__ == "__main__":
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_1 = run_experiment(m1, m2, m3, 0.1, 100000)
    c_05 = run_experiment(m1, m2, m3, 0.05, 100000)
    c_01 = run_experiment(m1, m2, m3, 0.01, 100000)
    figure()
    plot(c_1, label="eps = 0.1")
    plot(c_05, label="eps = 0.05")
    plot(c_01, label="eps = 0.01")
    legend()
    xscale("log")
    figure()
    plot(c_1, label="eps = 0.1")
    plot(c_05, label="eps = 0.05")
    plot(c_01, label="eps = 0.01")
    legend()
    show()
