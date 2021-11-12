"""
utility script
"""
from types import SimpleNamespace
from os.path import exists
from sys import exit as sexit
from numpy import (
    min as nmin,
    max as nmax,
    argmax,
    mean,
    vstack,
    array,
    linspace,
    empty,
    place,
    zeros,
    full,
    log,
    exp,
    cos,
    sin,
    pi,
    sqrt,
    float32,
    int32,
)
from numpy.random import randn, shuffle
from pandas import DataFrame, read_csv
from matplotlib.pyplot import show, plot, figure
from sklearn.decomposition import PCA


def get_clouds():
    """
    get clouds function
    """
    n_class = 500
    d_0 = 2
    x_1 = randn(n_class, d_0) + array([0, -2])
    x_2 = randn(n_class, d_0) + array([2, 2])
    x_3 = randn(n_class, d_0) + array([-2, 2])
    x_0 = vstack([x_1, x_2, x_3])
    y_0 = array([0] * n_class + [1] * n_class + [2] * n_class)
    return x_0, y_0


def get_spiral():
    """
    get spiral function
    """
    radius = linspace(1, 10, 100)
    thetas = empty((6, 100))
    for i in range(6):
        start_angle = pi * i / 3.0
        end_angle = start_angle + pi / 2
        points = linspace(start_angle, end_angle, 100)
        thetas[i] = points
    x_1 = empty((6, 100))
    x_2 = empty((6, 100))
    for i in range(6):
        x_1[i] = radius * cos(thetas[i])
        x_2[i] = radius * sin(thetas[i])
    x_0 = empty((600, 2))
    x_0[:, 0] = x_1.flatten()
    x_0[:, 1] = x_2.flatten()
    x_0 += randn(600, 2) * 0.5
    y_0 = array([0] * 100 + [1] * 100 + [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100)
    return x_0, y_0


def get_transformed_data():
    """
    get transformed data function
    """
    print("Reading in and transforming data...")
    if not exists("train.csv"):
        print("Looking for train.csv")
        print(
            "You have not downloaded the data and/or not placed the files in the correct location."
        )
        print("Please get the data from: https://www.kaggle.com/c/digit-recognizer")
        print("Place train.csv in the folder large_files adjacent to the class folder")
        sexit()
    d_f = DataFrame(read_csv("train.csv"))
    data = d_f.values.astype(float32)
    shuffle(data)
    x_0 = data[:, 1:]
    y_0 = data[:, 0].astype(int32)
    x_train = x_0[:-1000]
    y_train = y_0[:-1000]
    x_test = x_0[-1000:]
    y_test = y_0[-1000:]
    m_u = x_train.mean(axis=0)
    x_train = x_train - m_u
    x_test = x_test - m_u
    pca = PCA()
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    x_train = x_train[:, :300]
    x_test = x_test[:, :300]
    m_u = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - m_u) / std
    x_test = (x_test - m_u) / std
    return x_train, x_test, y_train, y_test


def get_normalized_data():
    """
    get normalized data function
    """
    print("Reading in and get normalized data...")
    if not exists("train.csv"):
        print("Looking for train.csv")
        print(
            "You have not downloaded the data and/or not placed the files in the correct location."
        )
        print("Please get the data from: https://www.kaggle.com/c/digit-recognizer")
        print("Place train.csv in the folder large_files adjacent to the class folder")
        sexit()
    d_f = DataFrame(read_csv("train.csv"))
    data = d_f.values.astype(float32)
    shuffle(data)
    x_0 = data[:, 1:]
    y_0 = data[:, 0]
    x_train = x_0[:-1000]
    y_train = y_0[:-1000]
    x_test = x_0[-1000:]
    y_test = y_0[-1000:]
    m_u = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    place(std, std == 0, 1)
    x_train = (x_train - m_u) / std
    x_test = (x_test - m_u) / std
    return x_train, x_test, y_train, y_test


def plot_cumulative_variance(pca):
    """
    plot cumulative variance function
    """
    p_0 = []
    for p_1 in pca.explained_variance_ratio_:
        if len(p_0) == 0:
            p_0.append(p_1)
        else:
            p_0.append(p_1 + p_0[-1])
    figure()
    plot(p_0)
    return p_0


def forward(x_0, w_0, b_0):
    """
    forward function
    """
    a_0 = x_0.dot(w_0) + b_0
    expa = exp(a_0)
    y_0 = expa / expa.sum(axis=1, keepdims=True)
    return y_0


def predict(p_y):
    """
    predict function
    """
    return argmax(p_y, axis=1)


def error_rate(p_y, t_0):
    """
    error rate function
    """
    prediction = predict(p_y)
    return mean(prediction != t_0)


def cost(p_y, t_0):
    """
    cost function
    """
    tot = t_0 * log(p_y)
    return -tot.sum()


def grad_w(t_0, y_0, x_0):
    """
    grad w function
    """
    return x_0.T.dot(t_0 - y_0)


def gradb(t_0, y_0):
    """
    gradb function
    """
    return (t_0 - y_0).sum(axis=0)


def y2indicator(y_0):
    """
    y2indicator function
    """
    n_0 = len(y_0)
    y_0 = y_0.astype(int32)
    ind = zeros((n_0, 10))
    for i in range(n_0):
        ind[i, y_0[i]] = 1
    return ind


def decorator(func):
    """
    decorator function
    """

    def wrapper():
        self = func()
        if self:
            for items in self.__dict__.items():
                print(items)
        return self

    return wrapper


@decorator
def benchmark_full(self=SimpleNamespace()):
    """
    benchmark full function
    """
    self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = get_normalized_data()
    print("Performing logistic regression...")
    _, self.D = self.Xtrain.shape
    self.Ytrain_ind = y2indicator(self.Ytrain)
    self.Ytest_ind = y2indicator(self.Ytest)
    self.W = randn(self.D, 10) / sqrt(self.D)
    self.b = zeros(10)
    self.LL = []
    self.LLtest = []
    self.CRtest = []
    self.lr = 0.00004
    self.reg = 0.01
    print(nmin(self.Xtrain), nmax(self.Xtrain), self.Xtrain.shape)
    print(nmin(self.W), nmax(self.W), self.W.shape)
    print(nmin(self.b), nmax(self.b), self.b.shape)
    for i in range(500):
        self.p_y = forward(self.Xtrain, self.W, self.b)
        self.ll = cost(self.p_y, self.Ytrain_ind)
        self.LL.append(self.ll)
        self.p_y_test = forward(self.Xtest, self.W, self.b)
        self.lltest = cost(self.p_y_test, self.Ytest_ind)
        self.LLtest.append(self.lltest)
        self.err = error_rate(self.p_y_test, self.Ytest)
        self.CRtest.append(self.err)
        self.W += self.lr * (
            grad_w(self.Ytrain_ind, self.p_y, self.Xtrain) - self.reg * self.W
        )
        self.b += self.lr * (gradb(self.Ytrain_ind, self.p_y) - self.reg * self.b)
        if i % 10 == 0:
            print(f"Cost at iteration {i}: {self.ll:.6f}")
            print("Error rate:", self.err)
    self.p_y = forward(self.Xtest, self.W, self.b)
    print("Final error rate:", error_rate(self.p_y, self.Ytest))
    self.iters = range(len(self.LL))
    figure()
    plot(self.iters, self.LL, self.iters, self.LLtest)
    figure()
    plot(self.CRtest)

    return self


@decorator
def benchmark_pca(self=SimpleNamespace()):
    """
    benchmark pca function
    """
    self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = get_transformed_data()
    print("Performing logistic regression...")
    self.N, self.D = self.Xtrain.shape
    self.Ytrain_ind = zeros((self.N, 10))
    for i in range(self.N):
        self.Ytrain_ind[i, self.Ytrain[i]] = 1
    self.Ntest = len(self.Ytest)
    self.Ytest_ind = zeros((self.Ntest, 10))
    for i in range(self.Ntest):
        self.Ytest_ind[i, self.Ytest[i]] = 1
    self.W = randn(self.D, 10) / sqrt(self.D)
    self.b = zeros(10)
    self.LL = []
    self.LLtest = []
    self.CRtest = []
    self.lr = 0.0001
    self.reg = 0.01
    for i in range(200):
        self.p_y = forward(self.Xtrain, self.W, self.b)
        self.ll = cost(self.p_y, self.Ytrain_ind)
        self.LL.append(self.ll)
        self.p_y_test = forward(self.Xtest, self.W, self.b)
        self.lltest = cost(self.p_y_test, self.Ytest_ind)
        self.LLtest.append(self.lltest)
        self.err = error_rate(self.p_y_test, self.Ytest)
        self.CRtest.append(self.err)
        self.W += self.lr * (
            grad_w(self.Ytrain_ind, self.p_y, self.Xtrain) - self.reg * self.W
        )
        self.b += self.lr * (gradb(self.Ytrain_ind, self.p_y) - self.reg * self.b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, self.ll))
            print("Error rate:", self.err)
    self.p_y = forward(self.Xtest, self.W, self.b)
    print("Final error rate:", error_rate(self.p_y, self.Ytest))
    self.iters = range(len(self.LL))
    figure()
    plot(self.iters, self.LL, self.iters, self.LLtest)
    figure()
    plot(self.CRtest)

    return self


if __name__ == "__main__":
    xcloud, ycloud = get_clouds()
    print(nmin(xcloud), nmax(xcloud), xcloud.shape)
    print(nmin(ycloud), nmax(ycloud), ycloud.shape)
    assert (-7 <= xcloud).all() and (xcloud <= 7).all()
    assert xcloud.shape == (1500, 2)
    assert (ycloud >= 0).all() and (ycloud <= 2).all()
    assert ycloud.shape == (1500,)

    xspiral, yspiral = get_spiral()
    print(nmin(xspiral), nmax(xspiral), xspiral.shape)
    print(nmin(yspiral), nmax(yspiral), yspiral.shape)
    assert (-12 <= xspiral).all() and (xspiral <= 12).all()
    assert xspiral.shape == (600, 2)
    assert (yspiral >= 0).all() and (yspiral <= 2).all()
    assert yspiral.shape == (600,)

    one, two, three, four = get_transformed_data()
    print(nmin(one), nmax(one), one.shape)
    print(nmin(two), nmax(two), two.shape)
    print(nmin(three), nmax(three), three.shape)
    print(nmin(four), nmax(four), four.shape)
    assert (one >= -9).all() and (one <= 9).all()
    assert (two >= -9).all() and (two <= 9).all()
    assert (three >= 0).all() and (three <= 9).all()
    assert (four >= 0).all() and (four <= 9).all()
    assert one.shape[0] == three.shape[0]
    assert one.shape[1] == two.shape[1]
    assert two.shape[0] == four.shape[0]
    assert len(three.shape) == len(four.shape)

    five, six, seven, eight = get_normalized_data()
    print(nmin(five), nmax(five), five.shape)
    print(nmin(six), nmax(six), six.shape)
    print(nmin(seven), nmax(seven), seven.shape)
    print(nmin(eight), nmax(eight), eight.shape)
    assert (five >= -1.3).all() and (five <= 210).all()
    assert (six >= -1.3).all() and (six <= 1000).all()
    assert (seven >= 0).all() and (seven <= 9).all()
    assert (eight >= 0).all() and (eight <= 9).all()
    assert five.shape[0] == seven.shape[0]
    assert five.shape[1] == six.shape[1]
    assert six.shape[0] == eight.shape[0]
    assert len(seven.shape) == len(eight.shape)

    pca_test = PCA()
    pca_test.fit_transform(full((41000, 784), 0.1))
    pca_test.transform(full((1000, 784), 0.2))
    pcv = plot_cumulative_variance(pca_test)
    print(nmin(pcv), nmax(pcv), len(pcv))
    assert (array(pcv) == full((784,), 1)).all()

    forw = forward(full((41000, 784), 0.1), full((784, 10), 0.2), full((10,), 0.3))
    print(nmin(forw), nmax(forw), forw.shape)
    assert (forw == full((41000, 10), 0.1)).all()

    show()
