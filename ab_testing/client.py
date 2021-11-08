"""
Bayesin Machine Learning in Python: A/B Testing
"""
from os.path import join, dirname
from requests import get, post
from pandas import DataFrame, read_csv


def main():
    """
    main function
    """
    df_var = DataFrame(read_csv(join(dirname(__file__), "advertisement_clicks.csv")))
    a_var = df_var[df_var["advertisement_id"] == "A"]
    b_var = df_var[df_var["advertisement_id"] == "B"]
    a_var = a_var["action"].values
    b_var = b_var["action"].values
    print("a.mean:", a_var.mean())
    print("b.mean:", b_var.mean())
    i = 0
    j = 0
    count = 0
    while i < len(a_var) and j < len(b_var):
        r_var = get("http://localhost:8888/get_ad")
        r_var = r_var.json()
        if r_var["advertisement_id"] == "A":
            action = a_var[i]
            i += 1
        else:
            action = b_var[j]
            j += 1
        if action == 1:
            post(
                "http://localhost:8888/click_ad",
                data={"advertisement_id": r_var["advertisement_id"]},
            )
        count += 1
        if count % 50 == 0:
            print("Seen %s ads, A: %s, B: %s" % (count, i, j))


if __name__ == "__main__":
    main()
