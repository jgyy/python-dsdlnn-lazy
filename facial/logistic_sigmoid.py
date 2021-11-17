"""
logistic sigmoid script
"""
from types import SimpleNamespace
from numpy import repeat, vstack, array, round as nround, sqrt
from numpy.random import randn
from matplotlib.pyplot import plot, show
from sklearn.utils import shuffle
from util import get_binary_data, sigmoid, sigmoid_cost, error_rate


class LogisticModel:
    """
    logistic model class
    """

    def __init__(self):
        self.w_data = 0
        self.b_data = 0

    def fit(self, x_data, y_data, show_fig=False):
        """
        fit method
        """
        sns = SimpleNamespace()
        sns.learning_rate = 1e-6
        sns.reg = 0.0
        sns.epochs = 12000
        x_data, y_data = shuffle(x_data, y_data)
        sns.Xvalid, sns.Yvalid = x_data[-1000:], y_data[-1000:]
        x_data, y_data = x_data[:-1000], y_data[:-1000]
        sns.N, sns.D = x_data.shape
        self.w_data = randn(sns.D) / sqrt(sns.D)
        self.b_data = 0
        sns.costs = []
        sns.best_validation_error = 1
        for i in range(sns.epochs):
            sns.pY = self.forward(x_data)
            self.w_data -= sns.learning_rate * (
                x_data.T.dot(sns.pY - y_data) + sns.reg * self.w_data
            )
            self.b_data -= sns.learning_rate * (
                (sns.pY - y_data).sum() + sns.reg * self.b_data
            )
            if i % 20 == 0:
                sns.pYvalid = self.forward(sns.Xvalid)
                sns.c = sigmoid_cost(sns.Yvalid, sns.pYvalid)
                sns.costs.append(sns.c)
                sns.e = error_rate(sns.Yvalid, nround(sns.pYvalid))
                print("i:", i, "cost:", sns.c, "error:", sns.e)
                if sns.e < sns.best_validation_error:
                    sns.best_validation_error = sns.e
        print("best_validation_error:", sns.best_validation_error)
        if show_fig:
            plot(sns.costs)

    def forward(self, x_data):
        """
        forward method
        """
        return sigmoid(x_data.dot(self.w_data) + self.b_data)

    def predict(self, x_data):
        """
        predict method
        """
        py_data = self.forward(x_data)
        return nround(py_data)

    def score(self, x_data, y_data):
        """
        score method
        """
        prediction = self.predict(x_data)
        return 1 - error_rate(y_data, prediction)


def main():
    """
    main function
    """
    xbin, ybin = get_binary_data()
    x_0 = xbin[ybin == 0, :]
    x_1 = xbin[ybin == 1, :]
    x_1 = repeat(x_1, 9, axis=0)
    xbin = vstack([x_0, x_1])
    ybin = array([0] * len(x_0) + [1] * len(x_1))
    model = LogisticModel()
    model.fit(xbin, ybin, show_fig=True)
    model.score(xbin, ybin)


if __name__ == "__main__":
    main()
    show()
