"""
Bayesin Machine Learning in Python: A/B Testing
"""
from numpy import float32, zeros, empty, ones
from numpy.random import random
from matplotlib.pyplot import plot, show
from scipy.stats import chi2


class DataGenerator:
    """
    Data Generator Class
    """

    def __init__(self, p_1, p_2):
        self.p_1 = p_1
        self.p_2 = p_2

    def __call__(self):
        print(self)

    def next(self):
        """
        next method
        """
        click1 = 1 if (random() < self.p_1) else 0
        click2 = 1 if (random() < self.p_2) else 0
        return click1, click2


def get_p_value(t_val):
    """
    get p value function
    """
    det = t_val[0, 0] * t_val[1, 1] - t_val[0, 1] * t_val[1, 0]
    c_2 = (
        float(det)
        / t_val[0].sum()
        * det
        / t_val[1].sum()
        * t_val.sum()
        / t_val[:, 0].sum()
        / t_val[:, 1].sum()
    )
    p_val = 1 - chi2.cdf(x=c_2, df=1)
    return p_val


def run_experiment(p_1, p_2, n_val):
    """
    run experiment function
    """
    data = DataGenerator(p_1, p_2)
    p_values = empty(n_val)
    t_val = zeros((2, 2)).astype(float32)
    for i in range(n_val):
        c_1, c_2 = data.next()
        t_val[0, c_1] += 1
        t_val[1, c_2] += 1
        if i < 10:
            p_values[i] = None
        else:
            p_values[i] = get_p_value(t_val)
    plot(p_values)
    plot(ones(n_val) * 0.05)


if __name__ == "__main__":
    run_experiment(0.1, 0.11, 20000)
    show()
