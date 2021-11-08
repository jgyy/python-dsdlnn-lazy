"""
From the course: Bayesin Machine Learning in Python: A/B Testing
"""
from os.path import join, dirname
from numpy import array
from pandas import DataFrame, read_csv
from scipy.stats import chi2


def get_p_value(t_var):
    """
    get p value function
    """
    det = t_var[0, 0] * t_var[1, 1] - t_var[0, 1] * t_var[1, 0]
    c2_var = (
        float(det)
        / t_var[0].sum()
        * det
        / t_var[1].sum()
        * t_var.sum()
        / t_var[:, 0].sum()
        / t_var[:, 1].sum()
    )
    p_var = 1 - chi2.cdf(x=c2_var, df=1)
    return p_var


def main():
    """
    main function
    """
    df_var = DataFrame(read_csv(join(dirname(__file__), "advertisement_clicks.csv")))
    a_var = df_var[df_var["advertisement_id"] == "A"]
    b_var = df_var[df_var["advertisement_id"] == "B"]
    a_var = a_var["action"]
    b_var = b_var["action"]
    a_clk = a_var.sum()
    a_noclk = a_var.size - a_var.sum()
    b_clk = b_var.sum()
    b_noclk = b_var.size - b_var.sum()
    t_var = array([[a_clk, a_noclk], [b_clk, b_noclk]])
    print(get_p_value(t_var))


if __name__ == "__main__":
    main()
