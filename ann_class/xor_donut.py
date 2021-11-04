"""
revisiting the XOR and donut problems to show how features
can be learned automatically using neural networks
"""
from numpy import (
    sum as nsum,
    round as nround,
    abs as nabs,
    outer,
    array,
    zeros,
    mean,
    concatenate,
    sin,
    cos,
    log,
    exp,
    pi,
)
from numpy.random import randn, random
from matplotlib.pyplot import show, plot, figure


def forward(x_var, w1_var, b1_var, w2_var, b2_var):
    """
    relu function
    """
    z_var = x_var.dot(w1_var) + b1_var
    z_var = z_var * (z_var > 0)

    activation = z_var.dot(w2_var) + b2_var
    y_var = 1 / (1 + exp(-activation))
    return y_var, z_var


def predict(x_var, w1_var, b1_var, w2_var, b2_var):
    """
    predict function
    """
    y_var, _ = forward(x_var, w1_var, b1_var, w2_var, b2_var)
    return nround(y_var)


def derivative_w2(z_var, t_var, y_var):
    """
    Z is (N, M)
    """
    return (t_var - y_var).dot(z_var)


def derivative_b2(t_var, y_var):
    """
    derivative b2 function
    """
    return (t_var - y_var).sum()


def derivative_w1(x_var, z_var, t_var, y_var, w2_var):
    """
    relu w1 activation
    """
    dz_var = outer(t_var - y_var, w2_var) * (z_var > 0)
    return x_var.T.dot(dz_var)


def derivative_b1(z_var, t_var, y_var, w2_var):
    """
    relu b1 activation
    """
    dz_var = outer(t_var - y_var, w2_var) * (z_var > 0)
    return dz_var.sum(axis=0)


def get_log_likelihood(t_var, y_var):
    """
    get log likelihood function
    """
    return nsum(t_var * log(y_var) + (1 - t_var) * log(1 - y_var))


def test_xor():
    """
    test xor function
    """
    xor = {
        "X": array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "Y": array([0, 1, 1, 0]),
        "W1": randn(2, 5),
        "b1": zeros(5),
        "W2": randn(5),
        "b2": 0,
        "LL": [],
        "learning_rate": 1e-2,
        "regularization": 0.0,
    }
    for i in range(30000):
        py_var, z_var = forward(xor["X"], xor["W1"], xor["b1"], xor["W2"], xor["b2"])
        ll_var = get_log_likelihood(xor["Y"], py_var)
        prediction = predict(xor["X"], xor["W1"], xor["b1"], xor["W2"], xor["b2"])
        xor["LL"].append(ll_var)
        gw2 = derivative_w2(z_var, xor["Y"], py_var)
        gb2 = derivative_b2(xor["Y"], py_var)
        gw1 = derivative_w1(xor["X"], z_var, xor["Y"], py_var, xor["W2"])
        gb1 = derivative_b1(z_var, xor["Y"], py_var, xor["W2"])
        xor["W2"] += xor["learning_rate"] * (gw2 - xor["regularization"] * xor["W2"])
        xor["b2"] += xor["learning_rate"] * (gb2 - xor["regularization"] * xor["b2"])
        xor["W1"] += xor["learning_rate"] * (gw1 - xor["regularization"] * xor["W1"])
        xor["b1"] += xor["learning_rate"] * (gb1 - xor["regularization"] * xor["b1"])
        if i % 1000 == 0:
            print(ll_var)
    print("final classification rate:", mean(prediction == xor["Y"]))

    figure()
    plot(xor["LL"])


def test_donut():
    """
    test donut example function
    """
    nut = {"N": 1000, "R_inner": 5, "R_outer": 10}
    nut |= {
        "R1": randn(nut["N"] // 2) + nut["R_inner"],
        "theta": lambda: 2 * pi * random(nut["N"] // 2),
    }
    nut |= {
        "X_inner": concatenate(
            [[nut["R1"] * cos(nut["theta"]())], [nut["R1"] * sin(nut["theta"]())]]
        ).T,
        "R2": randn(nut["N"] // 2) + nut["R_outer"],
    }
    nut |= {
        "X_outer": concatenate(
            [[nut["R2"] * cos(nut["theta"]())], [nut["R2"] * sin(nut["theta"]())]]
        ).T
    }
    nut |= {
        "X": concatenate([nut["X_inner"], nut["X_outer"]]),
        "Y": array([0] * (nut["N"] // 2) + [1] * (nut["N"] // 2)),
        "n_hidden": 8,
    }
    nut |= {
        "W1": randn(2, nut["n_hidden"]),
        "b1": randn(nut["n_hidden"]),
        "W2": randn(nut["n_hidden"]),
        "b2": randn(1),
        "LL": [],
        "learning_rate": 0.00005,
        "regularization": 0.2,
    }
    for i in range(3000):
        py_var, z_var = forward(nut["X"], nut["W1"], nut["b1"], nut["W2"], nut["b2"])
        ll_var = get_log_likelihood(nut["Y"], py_var)
        prediction = predict(nut["X"], nut["W1"], nut["b1"], nut["W2"], nut["b2"])
        er_var = nabs(prediction - nut["Y"]).mean()
        nut["LL"].append(ll_var)
        gw2 = derivative_w2(z_var, nut["Y"], py_var)
        gb2 = derivative_b2(nut["Y"], py_var)
        gw1 = derivative_w1(nut["X"], z_var, nut["Y"], py_var, nut["W2"])
        gb1 = derivative_b1(z_var, nut["Y"], py_var, nut["W2"])
        nut["W2"] += nut["learning_rate"] * (gw2 - nut["regularization"] * nut["W2"])
        nut["b2"] += nut["learning_rate"] * (gb2 - nut["regularization"] * nut["b2"])
        nut["W1"] += nut["learning_rate"] * (gw1 - nut["regularization"] * nut["W1"])
        nut["b1"] += nut["learning_rate"] * (gb1 - nut["regularization"] * nut["b1"])
        if i % 300 == 0:
            print("i:", i, "ll:", ll_var, "classification rate:", 1 - er_var)

    figure()
    plot(nut["LL"])


if __name__ == "__main__":
    test_xor()
    test_donut()
    show()
