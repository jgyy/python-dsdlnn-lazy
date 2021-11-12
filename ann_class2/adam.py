"""
Compare RMSprop with momentum vs. Adam
"""
from types import SimpleNamespace
from numpy import sqrt, zeros
from numpy.random import randn
from matplotlib.pyplot import plot, show, legend
from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b2, derivative_b1


def decorator(func):
    """
    decorator function
    """

    def wrapper(self):
        self = func(self)
        if self:
            for items in self.__dict__.items():
                print(items)
        show()

    return wrapper


@decorator
def main(self):
    """
    main function
    """
    self.max_iter = 10
    self.print_period = 10
    self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = get_normalized_data()
    self.reg = 0.01
    self.Ytrain_ind = y2indicator(self.Ytrain)
    self.Ytest_ind = y2indicator(self.Ytest)
    self.N, self.D = self.Xtrain.shape
    self.batch_sz = 500
    self.n_batches = self.N // self.batch_sz
    self.M = 300
    self.K = 10
    self.W1_0 = randn(self.D, self.M) / sqrt(self.D)
    self.b1_0 = zeros(self.M)
    self.W2_0 = randn(self.M, self.K) / sqrt(self.M)
    self.b2_0 = zeros(self.K)
    self.W1 = self.W1_0.copy()
    self.b1 = self.b1_0.copy()
    self.W2 = self.W2_0.copy()
    self.b2 = self.b2_0.copy()
    self.mW1 = 0
    self.mb1 = 0
    self.mW2 = 0
    self.mb2 = 0
    self.vW1 = 0
    self.vb1 = 0
    self.vW2 = 0
    self.vb2 = 0
    self.lr0 = 0.001
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.eps = 1e-8
    self.loss_adam = []
    self.err_adam = []
    self.t = 1

    self = iteration1(self)
    self = iteration2(self)
    return self


def iteration1(self):
    """
    iteration one function
    """
    for i in range(self.max_iter):
        for j in range(self.n_batches):
            self.Xbatch = self.Xtrain[
                j * self.batch_sz : (j * self.batch_sz + self.batch_sz),
            ]
            self.Ybatch = self.Ytrain_ind[
                j * self.batch_sz : (j * self.batch_sz + self.batch_sz),
            ]
            self.pYbatch, self.Z = forward(
                self.Xbatch, self.W1, self.b1, self.W2, self.b2
            )
            self.gW2 = (
                derivative_w2(self.Z, self.Ybatch, self.pYbatch) + self.reg * self.W2
            )
            self.gb2 = derivative_b2(self.Ybatch, self.pYbatch) + self.reg * self.b2
            self.gW1 = (
                derivative_w1(self.Xbatch, self.Z, self.Ybatch, self.pYbatch, self.W2)
                + self.reg * self.W1
            )
            self.gb1 = (
                derivative_b1(self.Z, self.Ybatch, self.pYbatch, self.W2)
                + self.reg * self.b1
            )
            self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * self.gW1
            self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * self.gb1
            self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * self.gW2
            self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * self.gb2
            self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * self.gW1 * self.gW1
            self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * self.gb1 * self.gb1
            self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * self.gW2 * self.gW2
            self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * self.gb2 * self.gb2
            self.correction1 = 1 - self.beta1 ** self.t
            self.hat_mW1 = self.mW1 / self.correction1
            self.hat_mb1 = self.mb1 / self.correction1
            self.hat_mW2 = self.mW2 / self.correction1
            self.hat_mb2 = self.mb2 / self.correction1
            self.correction2 = 1 - self.beta2 ** self.t
            self.hat_vW1 = self.vW1 / self.correction2
            self.hat_vb1 = self.vb1 / self.correction2
            self.hat_vW2 = self.vW2 / self.correction2
            self.hat_vb2 = self.vb2 / self.correction2
            self.t += 1
            self.W1 = self.W1 - self.lr0 * self.hat_mW1 / (
                sqrt(self.hat_vW1) + self.eps
            )
            self.b1 = self.b1 - self.lr0 * self.hat_mb1 / (
                sqrt(self.hat_vb1) + self.eps
            )
            self.W2 = self.W2 - self.lr0 * self.hat_mW2 / (
                sqrt(self.hat_vW2) + self.eps
            )
            self.b2 = self.b2 - self.lr0 * self.hat_mb2 / (
                sqrt(self.hat_vb2) + self.eps
            )
            if j % self.print_period == 0:
                self.pY, _ = forward(self.Xtest, self.W1, self.b1, self.W2, self.b2)
                self.l = cost(self.pY, self.Ytest_ind)
                self.loss_adam.append(self.l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, self.l))
                self.err = error_rate(self.pY, self.Ytest)
                self.err_adam.append(self.err)
                print("Error rate:", self.err)

    return self


def iteration2(self):
    """
    iteration one function
    """
    self.pY, _ = forward(self.Xtest, self.W1, self.b1, self.W2, self.b2)
    print("Final error rate:", error_rate(self.pY, self.Ytest))
    self.W1 = self.W1_0.copy()
    self.b1 = self.b1_0.copy()
    self.W2 = self.W2_0.copy()
    self.b2 = self.b2_0.copy()
    self.loss_rms = []
    self.err_rms = []
    self.lr0 = 0.001
    self.mu = 0.9
    self.decay_rate = 0.999
    self.eps = 1e-8
    self.cache_W2 = 1
    self.cache_b2 = 1
    self.cache_W1 = 1
    self.cache_b1 = 1
    self.dW1, self.db1, self.dW2, self.db2 = 0, 0, 0, 0
    for i in range(self.max_iter):
        for j in range(self.n_batches):
            self.Xbatch = self.Xtrain[
                j * self.batch_sz : (j * self.batch_sz + self.batch_sz),
            ]
            self.Ybatch = self.Ytrain_ind[
                j * self.batch_sz : (j * self.batch_sz + self.batch_sz),
            ]
            self.pYbatch, self.Z = forward(
                self.Xbatch, self.W1, self.b1, self.W2, self.b2
            )
            self.gW2 = (
                derivative_w2(self.Z, self.Ybatch, self.pYbatch) + self.reg * self.W2
            )
            self.gb2 = derivative_b2(self.Ybatch, self.pYbatch) + self.reg * self.b2
            self.gW1 = (
                derivative_w1(self.Xbatch, self.Z, self.Ybatch, self.pYbatch, self.W2)
                + self.reg * self.W1
            )
            self.gb1 = (
                derivative_b1(self.Z, self.Ybatch, self.pYbatch, self.W2)
                + self.reg * self.b1
            )
            self.cache_W2 = (
                self.decay_rate * self.cache_W2
                + (1 - self.decay_rate) * self.gW2 * self.gW2
            )
            self.cache_b2 = (
                self.decay_rate * self.cache_b2
                + (1 - self.decay_rate) * self.gb2 * self.gb2
            )
            self.cache_W1 = (
                self.decay_rate * self.cache_W1
                + (1 - self.decay_rate) * self.gW1 * self.gW1
            )
            self.cache_b1 = (
                self.decay_rate * self.cache_b1
                + (1 - self.decay_rate) * self.gb1 * self.gb1
            )
            self.dW2 = self.mu * self.dW2 + (1 - self.mu) * self.lr0 * self.gW2 / (
                sqrt(self.cache_W2) + self.eps
            )
            self.db2 = self.mu * self.db2 + (1 - self.mu) * self.lr0 * self.gb2 / (
                sqrt(self.cache_b2) + self.eps
            )
            self.dW1 = self.mu * self.dW1 + (1 - self.mu) * self.lr0 * self.gW1 / (
                sqrt(self.cache_W1) + self.eps
            )
            self.db1 = self.mu * self.db1 + (1 - self.mu) * self.lr0 * self.gb1 / (
                sqrt(self.cache_b1) + self.eps
            )
            self.W2 -= self.dW2
            self.b2 -= self.db2
            self.W1 -= self.dW1
            self.b1 -= self.db1
            if j % self.print_period == 0:
                self.pY, _ = forward(self.Xtest, self.W1, self.b1, self.W2, self.b2)
                self.l = cost(self.pY, self.Ytest_ind)
                self.loss_rms.append(self.l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, self.l))
                self.err = error_rate(self.pY, self.Ytest)
                self.err_rms.append(self.err)
                print("Error rate:", self.err)
    self.pY, _ = forward(self.Xtest, self.W1, self.b1, self.W2, self.b2)
    print("Final error rate:", error_rate(self.pY, self.Ytest))
    plot(self.loss_adam, label="adam")
    plot(self.loss_rms, label="rmsprop")
    legend()

    return self


if __name__ == "__main__":
    sns = SimpleNamespace()
    main(sns)
