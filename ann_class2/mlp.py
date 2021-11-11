"""
Simple multi-layer preceptron / neural network in Python and Numpy
"""
from numpy import exp, full, around


def forward(x_0, w_1, b_1, w_2, b_2):
    """
    forward script
    """
    z_0 = x_0.dot(w_1) + b_1
    z_0[z_0 < 0] = 0

    a_0 = z_0.dot(w_2) + b_2
    exp_a = exp(a_0)
    y_0 = exp_a / exp_a.sum(axis=1, keepdims=True)
    return y_0, z_0


def derivative_w2(z_0, t_0, y_0):
    """
    derivative w2 function
    """
    return z_0.T.dot(y_0 - t_0)


def derivative_b2(t_0, y_0):
    """
    derivative b2 function
    """
    return (y_0 - t_0).sum(axis=0)


def derivative_w1(x_0, z_0, t_0, y_0, w_2):
    """
    derivative w1 function
    """
    return x_0.T.dot(((y_0 - t_0).dot(w_2.T) * (z_0 > 0)))


def derivative_b1(z_0, t_0, y_0, w_2):
    """
    derivative b1 function
    """
    return ((y_0 - t_0).dot(w_2.T) * (z_0 > 0)).sum(axis=0)


if __name__ == "__main__":
    py_batch, z00 = forward(
        full((500, 784), 0.1),
        full((784, 300), 0.1),
        full((300,), 0.1),
        full((300, 10), 0.1),
        full((10,), 0.1),
    )
    assert (around(py_batch, 2) == full((500, 10), 0.1)).all()
    assert (around(z00, 2) == full((500, 300), 7.94)).all()
    w20 = derivative_w2(
        full((500, 300), 0.1), full((500, 10), 0.2), full((500, 10), 0.3)
    )
    print(w20, w20.shape)
    assert (around(w20, 2) == full((300, 10), 5)).all()
