"""
linear regression script
"""
from os.path import join, dirname
from numpy import nan, zeros, empty, full, concatenate
from pandas import DataFrame, read_csv
from matplotlib.pyplot import plot, show, figure
from sklearn.linear_model import LinearRegression


def main():
    """
    main function
    """
    df_var = DataFrame(
        read_csv(
            join(dirname(__file__), "international-airline-passengers.csv"),
            engine="python",
            skipfooter=3,
        )
    )
    df_var.columns = ["month", "num_passengers"]
    figure()
    plot(df_var.num_passengers)
    series = df_var.num_passengers.to_numpy()
    nums = len(series)
    for datas in (2, 3, 4, 5, 6, 7):
        num = nums - datas
        x_var = zeros((num, datas))
        for data in range(datas):
            x_var[:, data] = series[data : data + num]
        y_var = series[datas : datas + num]
        print("series length:", num)
        x_train = x_var[: int(num / 2)]
        y_train = y_var[: int(num / 2)]
        x_test = x_var[int(num / 2) :]
        y_test = y_var[int(num / 2) :]
        model = LinearRegression()
        model.fit(x_train, y_train)
        print("train score:", model.score(x_train, y_train))
        print("test score:", model.score(x_test, y_test))
        figure()
        plot(series)
        train_series = empty(num)
        train_series[: int(num / 2)] = model.predict(x_train)
        train_series[int(num / 2) :] = nan
        plot(concatenate([full(data, nan), train_series]))
        test_series = empty(num)
        test_series[: int(num / 2)] = nan
        test_series[int(num / 2) :] = model.predict(x_test)
        plot(concatenate([full(data, nan), test_series]))


if __name__ == "__main__":
    main()
    show()
