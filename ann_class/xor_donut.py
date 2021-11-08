"""
revisiting the XOR and donut problems to show how features
can be learned automatically using neural networks.
"""
from types import SimpleNamespace
from matplotlib.pyplot import show, plot
from numpy.random import randn, random
from numpy import (
    sum as nsum,
    round as nround,
    abs as nabs,
    concatenate,
    outer,
    array,
    zeros,
    mean,
    cos,
    sin,
    exp,
    log,
    pi,
)


def decorator(func):
    """
    decorator function
    """

    def wrapper():
        self = func()
        if self:
            for items in self.__dict__.items():
                print(items)
        show()

    return wrapper


def forward(self):
    """
    for binary classification! no softmax here
    """
    z_var = self.X.dot(self.W1) + self.b1
    z_var = z_var * (z_var > 0)

    activation = z_var.dot(self.W2) + self.b2
    y_var = 1 / (1 + exp(-activation))
    return y_var, z_var


def predict(self):
    """
    predict function
    """
    y_var, _ = forward(self)
    return nround(y_var)


def derivative_w2(self):
    """
    derivative w2 function
    """
    return (self.Y - self.pY).dot(self.Z)


def derivative_b2(self):
    """
    derivative b2 function
    """
    return (self.Y - self.pY).sum()


def derivative_w1(self):
    """
    derivative w1 function
    """
    d_z = outer(self.Y - self.pY, self.W2) * (self.Z > 0)
    return self.X.T.dot(d_z)


def derivative_b1(self):
    """
    derivative b1 function
    """
    d_z = outer(self.Y - self.pY, self.W2) * (self.Z > 0)
    return d_z.sum(axis=0)


def get_log_likelihood(self):
    """
    get log likelyhood function
    """
    return nsum(self.Y * log(self.pY) + (1 - self.Y) * log(1 - self.pY))


@decorator
def test_xor(self=SimpleNamespace()):
    """
    test xor function
    """
    self.X = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    self.Y = array([0, 1, 1, 0])
    self.W1 = randn(2, 5)
    self.b1 = zeros(5)
    self.W2 = randn(5)
    self.b2 = 0
    self.LL = []
    learning_rate = 1e-2
    regularization = 0.0
    for i in range(30000):
        self.pY, self.Z = forward(self)
        self.ll = get_log_likelihood(self)
        prediction = predict(self)
        self.LL.append(self.ll)
        gw2 = derivative_w2(self)
        gb2 = derivative_b2(self)
        gw1 = derivative_w1(self)
        gb1 = derivative_b1(self)
        self.W2 += learning_rate * (gw2 - regularization * self.W2)
        self.b2 += learning_rate * (gb2 - regularization * self.b2)
        self.W1 += learning_rate * (gw1 - regularization * self.W1)
        self.b1 += learning_rate * (gb1 - regularization * self.b1)
        if i % 1000 == 0:
            print(self.ll)
    print("final classification rate:", mean(prediction == self.Y))
    plot(self.LL)


@decorator
def test_donut(self=SimpleNamespace()):
    """
    test donut function
    """
    self.N = 1000
    r_inner = 5
    r_outer = 10
    self.R1 = randn(self.N // 2) + r_inner
    theta = 2 * pi * random(self.N // 2)
    x_inner = concatenate([[self.R1 * cos(theta)], [self.R1 * sin(theta)]]).T
    self.R2 = randn(self.N // 2) + r_outer
    theta = 2 * pi * random(self.N // 2)
    x_outer = concatenate([[self.R2 * cos(theta)], [self.R2 * sin(theta)]]).T
    self.X = concatenate([x_inner, x_outer])
    self.Y = array([0] * (self.N // 2) + [1] * (self.N // 2))
    n_hidden = 8
    self.W1 = randn(2, n_hidden)
    self.b1 = randn(n_hidden)
    self.W2 = randn(n_hidden)
    self.b2 = randn(1)
    self.LL = []
    learning_rate = 0.00005
    regularization = 0.2
    for i in range(3000):
        self.pY, self.Z = forward(self)
        self.ll = get_log_likelihood(self)
        prediction = predict(self)
        self.er = nabs(prediction - self.Y).mean()
        self.LL.append(self.ll)
        gw2 = derivative_w2(self)
        gb2 = derivative_b2(self)
        gw1 = derivative_w1(self)
        gb1 = derivative_b1(self)
        self.W2 += learning_rate * (gw2 - regularization * self.W2)
        self.b2 += learning_rate * (gb2 - regularization * self.b2)
        self.W1 += learning_rate * (gw1 - regularization * self.W1)
        self.b1 += learning_rate * (gb1 - regularization * self.b1)
        if i % 300 == 0:
            print("i:", i, "ll:", self.ll, "classification rate:", 1 - self.er)
    plot(self.LL)


if __name__ == "__main__":
    test_xor()
    test_donut()
