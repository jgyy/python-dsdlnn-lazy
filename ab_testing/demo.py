"""
Bayesin Machine Learning in Python: A/B Testing
"""
from numpy import linspace
from numpy.random import random
from matplotlib.pyplot import plot, title, show, figure
from scipy.stats import beta


def plot_func(a_var, b_var, trial, ctr):
    """
    plot function
    """
    x_var = linspace(0, 1, 200)
    y_var = beta.pdf(x_var, a_var, b_var)
    mean = float(a_var) / (a_var + b_var)
    figure()
    plot(x_var, y_var)
    title(
        "Distributions after %s trials, true rate = %.1f, mean = %.2f"
        % (trial, ctr, mean)
    )


def main():
    """
    main funtion
    """
    true_ctr = 0.3
    a_var, b_var = 1, 1
    show_list = [0, 5, 10, 25, 50, 100, 200, 300, 500, 700, 1000, 1500]
    for t_var in range(1501):
        coin_toss_result = random() < true_ctr
        if coin_toss_result:
            a_var += 1
        else:
            b_var += 1
        if t_var in show_list:
            plot_func(a_var, b_var, t_var + 1, true_ctr)


if __name__ == "__main__":
    main()
    show()
