"""
From the course: Bayesin Machine Learning in Python: A/B Testing
"""
from matplotlib.pyplot import plot, show, legend, title, figure
from numpy import sqrt, empty, linspace
from numpy.random import random
from scipy.stats import beta, norm


def main():
    """
    main function
    """
    t_val = 501
    true_ctr = 0.5
    a_val, b_val = 1, 1
    plot_indices = (10, 20, 30, 50, 100, 200, 500)
    data = empty(t_val)
    for i in range(t_val):
        x_val = 1 if random() < true_ctr else 0
        data[i] = x_val
        a_val += x_val
        b_val += 1 - x_val
        if i in plot_indices:
            p_val = data[:i].mean()
            n_val = i + 1
            std = sqrt(p_val * (1 - p_val) / n_val)
            x_val = linspace(0, 1, 200)
            g_val = norm.pdf(x_val, loc=p_val, scale=std)
            figure()
            plot(x_val, g_val, label="Gaussian Approximation")
            posterior = beta.pdf(x_val, a=a_val, b=b_val)
            plot(x_val, posterior, label="Beta Posterior")
            legend()
            title("N = %s" % n_val)


if __name__ == "__main__":
    main()
    show()
