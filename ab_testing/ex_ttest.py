"""
From the course: Bayesin Machine Learning in Python: A/B Testing
"""
from os.path import join, dirname
from numpy import abs as nabs, sqrt
from pandas import DataFrame, read_csv
from scipy import stats


def main():
    """
    main function
    """
    df_var = DataFrame(read_csv(join(dirname(__file__), "advertisement_clicks.csv")))
    a_var = df_var[df_var["advertisement_id"] == "A"]
    b_var = df_var[df_var["advertisement_id"] == "B"]
    a_var = a_var["action"]
    b_var = b_var["action"]
    print("a.mean:", a_var.mean())
    print("b.mean:", b_var.mean())
    t_var, p_var = stats.ttest_ind(a_var, b_var)
    print("t:\t", t_var, "p:\t", p_var)
    t_var, p_var = stats.ttest_ind(a_var, b_var, equal_var=False)
    print("Welch's t-test:")
    print("t:\t", t_var, "p:\t", p_var)
    n1_var = len(a_var)
    s1_sq = a_var.var()
    n2_var = len(b_var)
    s2_sq = b_var.var()
    t_var = (a_var.mean() - b_var.mean()) / sqrt(s1_sq / n1_var + s2_sq / n2_var)
    nu1 = n1_var - 1
    nu2 = n2_var - 1
    df_var = (s1_sq / n1_var + s2_sq / n2_var) ** 2 / (
        (s1_sq * s1_sq) / (n1_var * n1_var * nu1)
        + (s2_sq * s2_sq) / (n2_var * n2_var * nu2)
    )
    p_var = (1 - stats.t.cdf(nabs(t_var), df=df_var)) * 2
    print("Manual Welch t-test")
    print("t:\t", t_var, "p:\t", p_var)


if __name__ == "__main__":
    main()
