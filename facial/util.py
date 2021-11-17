"""
util script
"""
from numpy import (
    zeros,
    prod,
    arange,
    array,
    mean,
    repeat,
    vstack,
    concatenate,
    exp,
    sqrt,
    float32,
    log,
)
from numpy.random import randn, randint
from sklearn.utils import shuffle


def init_weight_and_bias(m_1, m_2):
    """
    init weight and bias function
    """
    w_0 = randn(m_1, m_2) / sqrt(m_1)
    b_0 = zeros(m_2)
    return w_0.astype(float32), b_0.astype(float32)


def init_filter(shape, poolsz):
    """
    init filter function
    """
    w_0 = (
        randn(*shape)
        * sqrt(2)
        / sqrt(prod(shape[1:]) + shape[0] * prod(shape[2:] / prod(poolsz)))
    )
    return w_0.astype(float32)


def relu(x_0):
    """
    relu function
    """
    return x_0 * (x_0 > 0)


def sigmoid(a_0):
    """
    sigmoid function
    """
    return 1 / (1 + exp(-a_0))


def softmax(a_0):
    """
    softmax function
    """
    exp_a = exp(a_0)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def sigmoid_cost(t_0, y_0):
    """
    sigmoid cost
    """
    return -(t_0 * log(y_0) + (1 - t_0) * log(1 - y_0)).sum()


def cost(t_0, y_0):
    """
    cost function
    """
    return -(t_0 * log(y_0)).sum()


def cost2(t_0, y_0):
    """
    cost 3 function
    """
    n_0 = len(t_0)
    return -log(y_0[arange(n_0), t_0]).mean()


def error_rate(targets, predictions):
    """
    error rate function
    """
    return mean(targets != predictions)


def y2indicator(y_0):
    """
    y2 indicator function
    """
    n_0 = len(y_0)
    k_0 = len(set(y_0))
    ind = zeros((n_0, k_0))
    for i in range(n_0):
        ind[i, y_0[i]] = 1
    return ind


def get_data(balance_ones=True, n_test=1000):
    """
    get data function
    """
    y_list = []
    x_list = []
    first = True
    with open("fer2013.csv", encoding="utf-8") as fer:
        for line in fer:
            if first:
                first = False
            else:
                row = line.split(",")
                y_list.append(int(row[0]))
                x_list.append([int(p) for p in row[1].split()])
    x_list, y_list = array(x_list) / 255.0, array(y_list)
    x_list, y_list = shuffle(x_list, y_list)
    x_train, y_train = x_list[:-n_test], y_list[:-n_test]
    x_valid, y_valid = x_list[-n_test:], y_list[-n_test:]
    if balance_ones:
        x_0, y_0 = x_train[y_train != 1, :], y_train[y_train != 1]
        x_1 = x_train[y_train == 1, :]
        x_1 = repeat(x_1, 9, axis=0)
        x_train = vstack([x_0, x_1])
        y_train = concatenate((y_0, [1] * len(x_1)))
    return x_train, y_train, x_valid, y_valid


def get_image_data():
    """
    get image data function
    """
    x_train, y_train, x_valid, y_valid = get_data()
    _, d_0 = x_train.shape
    d_0 = int(sqrt(d_0))
    x_train = x_train.reshape(-1, 1, d_0, d_0)
    x_valid = x_valid.reshape(-1, 1, d_0, d_0)
    return x_train, y_train, x_valid, y_valid


def get_binary_data():
    """
    get binary data function
    """
    y_list = []
    x_list = []
    first = True
    with open("fer2013.csv", encoding="utf-8") as fer:
        for line in fer:
            if first:
                first = False
            else:
                row = line.split(",")
                y_int = int(row[0])
                if y_int in (0, 1):
                    y_list.append(y_int)
                    x_list.append([int(p) for p in row[1].split()])
    return array(x_list) / 255.0, array(y_list)


def cross_validation(model, x_0, y_0, k_0=5):
    """
    cross validation function
    """
    x_0, y_0 = shuffle(x_0, y_0)
    s_z = len(y_0) // k_0
    errors = []
    for k in range(k_0):
        xtr = concatenate([x_0[: k * s_z, :], x_0[(k * s_z + s_z) :, :]])
        ytr = concatenate([y_0[: k * s_z], y_0[(k * s_z + s_z) :]])
        xte = x_0[k * s_z : (k * s_z + s_z), :]
        yte = y_0[k * s_z : (k * s_z + s_z)]
        model.fit(xtr, ytr)
        err = model.score(xte, yte)
        errors.append(err)
    print("errors:", errors)
    return mean(errors)


if __name__ == "__main__":
    gbdx, gbdy = get_binary_data()
    print(gbdx.min(), gbdx.max(), gbdx.shape)
    print(gbdy.min(), gbdy.max(), gbdy.shape)
    assert (gbdx >= 0).all() and (gbdx <= 1).all() and gbdx.shape == (5500, 2304)
    assert (gbdy >= 0).all() and (gbdy <= 1).all() and gbdy.shape == (5500,)

    sig = sigmoid(randint(-100, 100, (8876,)) / 100)
    print(sig.min(), sig.max(), sig.shape)
    assert (sig >= 0.2).all() and (sig <= 0.8).all() and sig.shape == (8876,)

    sigc = sigmoid_cost(randint(1, 100, (1000,)) / 100, randint(1, 100, (1000,)) / 100)
    print(sigc)
    assert 910 <= sigc <= 1010

    errr = error_rate(randint(0, 100, (1000,)) / 100, randint(30, 70, (1000,)) / 100)
    print(errr)
    assert 0.9 <= errr <= 1

    rel = relu(randint(-300, 300, (8876, 100)) / 100)
    print(rel.min(), rel.max(), rel.shape)
    assert (rel >= 0).all() and (rel <= 3).all() and rel.shape == (8876, 100)

    one, two, three, four = get_data()
    print(one.min(), one.max(), one.shape)
    print(two.min(), two.max(), two.shape)
    print(three.min(), three.max(), three.shape)
    print(four.min(), four.max(), four.shape)
    assert (one >= 0).all() and (one <= 1).all()
    assert (two >= 0).all() and (two <= 6).all()
    assert (three >= 0).all() and (three <= 1).all()
    assert (four >= 0).all() and (four <= 6).all()
    assert one.shape[0] == two.shape[0]
    assert one.shape[1] == three.shape[1]
    assert three.shape[0] == four.shape[0]

    five, six, seven, eight = get_image_data()
    print(five.min(), five.max(), five.shape)
    print(six.min(), six.max(), six.shape)
    print(seven.min(), seven.max(), seven.shape)
    print(eight.min(), eight.max(), eight.shape)
    assert (five >= 0).all() and (five <= 1).all()
    assert (six >= 0).all() and (six <= 6).all()
    assert (seven >= 0).all() and (seven <= 1).all()
    assert (eight >= 0).all() and (eight <= 6).all()
    assert five.shape[0] == six.shape[0]
    assert five.shape[1:] == seven.shape[1:]
    assert seven.shape[0] == eight.shape[0]

    y_2 = y2indicator(randint(0, 6, (39159,)))
    print(y_2.min(), y_2.max(), y_2.shape)
    assert (y_2 >= 0).all() and (y_2 <= 1).all() and y_2.shape == (39159, 6)

    w00, b00 = init_weight_and_bias(2304, 2000)
    print(w00.min(), w00.max(), w00.shape)
    print(b00.min(), b00.max(), b00.shape)
    assert (w00 >= -0.2).all() and (w00 <= 0.2).all() and w00.shape == (2304, 2000)
    assert (b00 >= 0).all() and b00.shape == (2000,)
