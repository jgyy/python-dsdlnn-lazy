"""
Bayesin Machine Learning in Python: A/B Testing
"""
from numpy import sqrt
from numpy.random import randn
from scipy import stats


def main():
    """
    main function
    """
    n_var = 10
    a_var = randn(n_var) + 2
    b_var = randn(n_var)
    var_a = a_var.var(ddof=1)
    var_b = b_var.var(ddof=1)
    s_var = sqrt((var_a + var_b) / 2)
    t_var = (a_var.mean() - b_var.mean()) / (s_var * sqrt(2.0 / n_var))
    df_var = 2 * n_var - 2
    p_var = 1 - stats.t.cdf(abs(t_var), df=df_var)
    print("t:\t", t_var, "p:\t", 2 * p_var)
    t2_var, p2_var = stats.ttest_ind(a_var, b_var)
    print("t2:\t", t2_var, "p2:\t", p2_var)


if __name__ == "__main__":
    main()
