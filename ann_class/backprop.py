"""
backpropagation example for deep learning in python class with sigmoid activation
"""
from types import SimpleNamespace
from numpy import exp, log, array, vstack, zeros, argmax
from numpy.random import randn
from matplotlib.pyplot import show, scatter, plot, figure


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
    forward function
    """
    if not any([self.X.any(), self.W1, self.b1, self.W2, self.b2]):
        raise Exception(self.X, self.W1, self.b1, self.W2, self.b2)
    z_var = 1 / (1 + exp(-self.X.dot(self.W1) - self.b1))
    a_var = z_var.dot(self.W2) + self.b2
    exp_a = exp(a_var)
    y_var = exp_a / exp_a.sum(axis=1, keepdims=True)
    return y_var, z_var


def classification_rate(self):
    """
    determine the classification rate (num correct / num total)
    """
    if not any([self.Y.any(), self.P]):
        raise Exception(self.Y, self.P)
    n_correct = 0
    n_total = 0
    for index, value in enumerate(self.Y):
        n_total += 1
        if value == self.P[index]:
            n_correct += 1
    return float(n_correct) / n_total


def derivative_w2(self):
    """
    derivative w2 function
    """
    if not any([self.T.any(), self.hidden, self.output]):
        raise Exception(self.T, self.hidden, self.output)
    ret4 = self.hidden.T.dot(self.T - self.output)
    return ret4


def derivative_w1(self):
    """
    derivative w1 function
    """
    if not any([self.T.any(), self.hidden, self.output, self.X, self.W2]):
        raise Exception(self.T, self.hidden, self.output, self.X, self.W2)
    d_z = (self.T - self.output).dot(self.W2.T) * self.hidden * (1 - self.hidden)
    ret2 = self.X.T.dot(d_z)
    return ret2


def derivative_b2(self):
    """
    derivative b2 function
    """
    if not any([self.T.any(), self.output]):
        raise Exception(self.T, self.output)
    return (self.T - self.output).sum(axis=0)


def derivative_b1(self):
    """
    derivative b1 function
    """
    if not any([self.T.any(), self.output, self.W2, self.hidden]):
        raise Exception(self.T, self.output, self.W2, self.hidden)
    return ((self.T - self.output).dot(self.W2.T) * self.hidden * (1 - self.hidden)).sum(axis=0)


def cost(self):
    """
    cost function
    """
    if not any([self.T.any(), self.output]):
        raise Exception(self.T, self.output)
    tot = self.T * log(self.output)
    return tot.sum()


@decorator
def main(self=SimpleNamespace()):
    """
    main function
    """
    self.Nclass = 500
    self.D = 2
    self.M = 3
    self.K = 3
    self.X1 = randn(self.Nclass, self.D) + array([0, -2])
    self.X2 = randn(self.Nclass, self.D) + array([2, 2])
    self.X3 = randn(self.Nclass, self.D) + array([-2, 2])
    self.X = vstack([self.X1, self.X2, self.X3])
    self.Y = array([0] * self.Nclass + [1] * self.Nclass + [2] * self.Nclass)
    self.N = len(self.Y)
    self.T = zeros((self.N, self.K))
    for i in range(self.N):
        self.T[i, self.Y[i]] = 1

    figure()
    scatter(self.X[:, 0], self.X[:, 1], c=self.Y, s=100, alpha=0.5)
    self.W1 = randn(self.D, self.M)
    self.b1 = randn(self.M)
    self.W2 = randn(self.M, self.K)
    self.b2 = randn(self.K)
    learning_rate = 1e-3
    costs = []
    for epoch in range(1000):
        self.output, self.hidden = forward(self)
        if epoch % 100 == 0:
            self.c = cost(self)
            self.P = argmax(self.output, axis=1)
            self.r = classification_rate(self)
            print("cost:", self.c, "classification_rate:", self.r)
            costs.append(self.c)
        gw2 = derivative_w2(self)
        gb2 = derivative_b2(self)
        gw1 = derivative_w1(self)
        gb1 = derivative_b1(self)
        self.W2 += learning_rate * gw2
        self.b2 += learning_rate * gb2
        self.W1 += learning_rate * gw1
        self.b1 += learning_rate * gb1

    figure()
    plot(costs)

    return self


if __name__ == "__main__":
    main()
