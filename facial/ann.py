"""
ann script
"""
from types import SimpleNamespace
from numpy import argmax, zeros, tanh, sqrt
from numpy.random import randn
from matplotlib.pyplot import plot, show
from util import get_data, softmax, cost2, y2indicator, error_rate


class ANN:
    """
    ann class
    """

    def __init__(self, M):
        self.m_0 = M
        self.w_1 = None
        self.b_1 = None
        self.w_2 = None
        self.b_2 = None

    def fit(self, x_0, y_0, x_valid, y_valid):
        """
        ann fit method
        """
        sns = SimpleNamespace()
        sns.learning_rate = 1e-6
        sns.reg = 0
        sns.epochs = 200
        sns.show_fig = True
        _, sns.D = x_0.shape
        sns.K = len(set(y_0))
        sns.T = y2indicator(y_0)
        self.w_1 = randn(sns.D, self.m_0) / sqrt(sns.D)
        self.b_1 = zeros(self.m_0)
        self.w_2 = randn(self.m_0, sns.K) / sqrt(self.m_0)
        self.b_2 = zeros(sns.K)
        sns.costs = []
        sns.best_validation_error = 1
        for i in range(sns.epochs):
            sns.pY, sns.Z = self.forward(x_0)
            sns.pY_T = sns.pY - sns.T
            self.w_2 -= sns.learning_rate * (sns.Z.T.dot(sns.pY_T) + sns.reg * self.w_2)
            self.b_2 -= sns.learning_rate * (sns.pY_T.sum(axis=0) + sns.reg * self.b_2)
            sns.dZ = sns.pY_T.dot(self.w_2.T) * (1 - sns.Z * sns.Z)
            self.w_1 -= sns.learning_rate * (x_0.T.dot(sns.dZ) + sns.reg * self.w_1)
            self.b_1 -= sns.learning_rate * (sns.dZ.sum(axis=0) + sns.reg * self.b_1)
            if i % 2 == 0:
                sns.pYvalid, _ = self.forward(x_valid)
                sns.c = cost2(y_valid, sns.pYvalid)
                sns.costs.append(sns.c)
                sns.e = error_rate(y_valid, argmax(sns.pYvalid, axis=1))
                print("i:", i, "cost:", sns.c, "error:", sns.e)
                if sns.e < sns.best_validation_error:
                    sns.best_validation_error = sns.e
        print("best_validation_error:", sns.best_validation_error)
        if sns.show_fig:
            plot(sns.costs)

    def forward(self, x_0):
        """
        ann forward method
        """
        z_0 = tanh(x_0.dot(self.w_1) + self.b_1)
        return softmax(z_0.dot(self.w_2) + self.b_2), z_0

    def predict(self, x_0):
        """
        ann predict method
        """
        p_y, _ = self.forward(x_0)
        return argmax(p_y, axis=1)

    def score(self, x_0, y_0):
        """
        ann score method
        """
        prediction = self.predict(x_0)
        return 1 - error_rate(y_0, prediction)


def main():
    """
    main function
    """
    x_train, y_train, x_valid, y_valid = get_data()
    model = ANN(200)
    model.fit(x_train, y_train, x_valid, y_valid)
    print(model.score(x_valid, y_valid))


if __name__ == "__main__":
    main()
    show()
