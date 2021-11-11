"""
process script
"""
from os.path import abspath, dirname, realpath, join
from numpy import zeros, int32
from numpy.random import shuffle
from pandas import DataFrame, read_csv


def get_data():
    """
    get data function
    """
    data = DataFrame(
        read_csv(join(abspath(dirname(realpath(__file__))), "ecommerce_data.csv"))
    ).values
    shuffle(data)
    x_data = data[:, :-1]
    y_data = data[:, -1].astype(int32)
    n_data, d_data = x_data.shape
    x_2 = zeros((n_data, d_data + 3))
    x_2[:, 0 : (d_data - 1)] = x_data[:, 0 : (d_data - 1)]
    for num in range(n_data):
        t_data = int(x_data[num, d_data - 1])
        x_2[num, t_data + d_data - 1] = 1
    x_data = x_2
    x_train = x_data[:-100]
    y_train = y_data[:-100]
    x_test = x_data[-100:]
    y_test = y_data[-100:]
    for i in (1, 2):
        m_data = x_train[:, i].mean()
        s_data = x_train[:, i].std()
        x_train[:, i] = (x_train[:, i] - m_data) / s_data
        x_test[:, i] = (x_test[:, i] - m_data) / s_data
    return x_train, y_train, x_test, y_test


def get_binary_data():
    """
    return only the data from the first 2 classes
    """
    x_train, y_train, x_test, y_test = get_data()
    x2_train = x_train[y_train <= 1]
    y2_train = y_train[y_train <= 1]
    x2_test = x_test[y_test <= 1]
    y2_test = y_test[y_test <= 1]
    return x2_train, y2_train, x2_test, y2_test


if __name__ == "__main__":
    one, two, three, four = get_data()
    print(one.shape, two.shape, three.shape, four.shape)
    assert one.shape == (400, 8)
    assert two.shape == (400,)
    assert three.shape == (100, 8)
    assert four.shape == (100,)
    one, two, three, four = get_binary_data()
