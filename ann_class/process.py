"""
process script
"""
from os.path import abspath, dirname, realpath, join
from numpy import min as nmin, max as nmax, zeros, int32
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
    print(nmin(one), nmax(one))
    print(nmin(two), nmax(two))
    print(nmin(three), nmax(three))
    print(nmin(four), nmax(four))
    print(one.shape, two.shape, three.shape, four.shape)
    assert (-2 <= one).all() and (one <= 6).all()
    assert (two >= 0).all() and (two <= 3).all()
    assert (-2 <= three).all() and (three <= 6).all()
    assert (four >= 0).all() and (four <= 3).all()
    assert one.shape[0] == two.shape[0]
    assert one.shape[1] == three.shape[1]
    assert three.shape[0] == four.shape[0]
    assert len(two.shape) == len(four.shape)

    five, six, seven, eight = get_binary_data()
    print(nmin(five), nmax(five))
    print(nmin(six), nmax(six))
    print(nmin(seven), nmax(seven))
    print(nmin(eight), nmax(eight))
    print(five.shape, six.shape, seven.shape, eight.shape)
    assert (-2 <= five).all() and (five <= 6).all()
    assert (six >= 0).all() and (six <= 1).all()
    assert (-2 <= seven).all() and (seven <= 6).all()
    assert (eight >= 0).all() and (eight <= 1).all()
    assert five.shape[0] == six.shape[0]
    assert five.shape[1] == seven.shape[1]
    assert seven.shape[0] == eight.shape[0]
    assert len(six.shape) == len(eight.shape)
