"""
ann sigmoid script
"""
from types import SimpleNamespace
from numpy import sqrt, outer, zeros, repeat, vstack, array
from numpy.random import randn
from matplotlib.pyplot import plot, show
from sklearn.utils import shuffle
from util import get_binary_data, sigmoid, sigmoid_cost, error_rate, relu


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

    def fit(self, x_0, y_0, show_fig=False):
        """
        ann fit method
        """
        sns = SimpleNamespace()
        sns.learning_rate = 5e-7
        sns.reg = 1.0
        sns.epochs = 200
        x_0, y_0 = shuffle(x_0, y_0)
        sns.Xvalid, sns.Yvalid = x_0[-1000:], y_0[-1000:]
        x_0, y_0 = x_0[:-1000], y_0[:-1000]
        sns.N, sns.D = x_0.shape
        self.w_1 = randn(sns.D, self.m_0) / sqrt(sns.D)
        self.b_1 = zeros(self.m_0)
        self.w_2 = randn(self.m_0) / sqrt(self.m_0)
        self.b_2 = 0
        sns.costs = []
        sns.best_validation_error = 1
        for i in range(sns.epochs):
            sns.pY, sns.Z = self.forward(x_0)
            sns.pY_Y = sns.pY - y_0
            self.w_2 -= sns.learning_rate * (sns.Z.T.dot(sns.pY_Y) + sns.reg * self.w_2)
            self.b_2 -= sns.learning_rate * ((sns.pY_Y).sum() + sns.reg * self.b_2)
            sns.dZ = outer(sns.pY_Y, self.w_2) * (sns.Z > 0)
            self.w_1 -= sns.learning_rate * (x_0.T.dot(sns.dZ) + sns.reg * self.w_1)
            self.b_1 -= sns.learning_rate * (sns.dZ.sum(axis=0) + sns.reg * self.b_1)
            if i % 2 == 0:
                sns.pYvalid, _ = self.forward(sns.Xvalid)
                sns.c = sigmoid_cost(sns.Yvalid, sns.pYvalid)
                sns.costs.append(sns.c)
                sns.e = error_rate(sns.Yvalid, sns.pYvalid.round())
                print("i:", i, "cost:", sns.c, "error:", sns.e)
                if sns.e < sns.best_validation_error:
                    sns.best_validation_error = sns.e
        print("best_validation_error:", sns.best_validation_error)
        if show_fig:
            plot(sns.costs)
            show()

    def forward(self, x_0):
        """
        ann forward method
        """
        print(
            (x_0.dot(self.w_1) + self.b_1).min(),
            (x_0.dot(self.w_1) + self.b_1).max(),
            (x_0.dot(self.w_1) + self.b_1).shape,
        )
        z_0 = relu(x_0.dot(self.w_1) + self.b_1)
        return sigmoid(z_0.dot(self.w_2) + self.b_2), z_0

    def predict(self, x_0):
        """
        ann predict method
        """
        p_0, y_0 = self.forward(x_0)
        return p_0.round(), y_0.round()

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
    x_data, y_data = get_binary_data()
    x0_data = x_data[y_data == 0, :]
    x1_data = x_data[y_data == 1, :]
    x1_data = repeat(x1_data, 9, axis=0)
    x_data = vstack([x0_data, x1_data])
    y_data = array([0] * len(x0_data) + [1] * len(x1_data))
    model = ANN(100)
    model.fit(x_data, y_data, show_fig=True)


if __name__ == "__main__":
    main()
    show()
