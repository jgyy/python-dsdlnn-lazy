"""
forward propagation example for deep learning in python class
"""
from numpy import vstack, array, argmax, exp
from numpy.random import randn
from matplotlib.pyplot import scatter, show


def sigmoid(a_var):
    """
    sigmoid function
    """
    return 1 / (1 + exp(-a_var))


def forward(x_var, w1_var, b1_var, w2_var, b2_var):
    """
    forward propagation function
    """
    z_var = sigmoid(x_var.dot(w1_var) + b1_var)
    a_var = z_var.dot(w2_var) + b2_var
    exp_a = exp(a_var)
    y_var = exp_a / exp_a.sum(axis=1, keepdims=True)
    return y_var


def classification_rate(y_var, p_var):
    """
    determine the classification rate (num correct / num total)
    """
    n_correct = 0
    n_total = 0
    for index, value in enumerate(y_var):
        n_total += 1
        if value == p_var[index]:
            n_correct += 1
    return float(n_correct) / n_total


def main():
    """
    main function
    """
    n_class = 500
    x1_var = randn(n_class, 2) + array([0, -2])
    x2_var = randn(n_class, 2) + array([2, 2])
    x3_var = randn(n_class, 2) + array([-2, 2])
    x_var = vstack([x1_var, x2_var, x3_var])
    y_var = array([0] * n_class + [1] * n_class + [2] * n_class)
    scatter(x_var[:, 0], x_var[:, 1], c=y_var, s=100, alpha=0.5)
    d_var = 2
    m_var = 3
    k_var = 3
    w1_var = randn(d_var, m_var)
    b1_var = randn(m_var)
    w2_var = randn(m_var, k_var)
    b2_var = randn(k_var)
    p_y_given_x = forward(x_var, w1_var, b1_var, w2_var, b2_var)
    p_var = argmax(p_y_given_x, axis=1)
    assert len(p_var) == len(y_var)
    print("Classification rate for randomly chosen weights:", classification_rate(y_var, p_var))


if __name__ == "__main__":
    main()
    show()
