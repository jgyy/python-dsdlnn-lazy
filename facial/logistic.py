"""
logistic script
"""
from types import SimpleNamespace
from numpy import zeros, argmax, sqrt
from numpy.random import randn
from matplotlib.pyplot import plot, show
from util import get_data, softmax, cost, y2indicator, error_rate


class LogisticModel:
    """
    logistic model class
    """

    def __init__(self):
        self.w_data = None
        self.b_data = None

    def fit(self, x_data, y_data, x_valid, y_valid):
        """
        fit method
        """
        sns = SimpleNamespace()
        sns.learning_rate = 1e-7
        sns.reg = 0.0
        sns.epochs = 1000
        sns.show_fig = True
        sns.Tvalid = y2indicator(y_valid)
        sns.N, sns.D = x_data.shape
        sns.K = len(set(y_data))
        sns.T = y2indicator(y_data)
        self.w_data = randn(sns.D, sns.K) / sqrt(sns.D)
        self.b_data = zeros(sns.K)
        sns.costs = []
        sns.best_validation_error = 1
        for i in range(sns.epochs):
            sns.pY = self.forward(x_data)
            self.w_data -= sns.learning_rate * (
                x_data.T.dot(sns.pY - sns.T) + sns.reg * self.w_data
            )
            self.b_data -= sns.learning_rate * (
                (sns.pY - sns.T).sum(axis=0) + sns.reg * self.b_data
            )
            if i % 10 == 0:
                sns.pYvalid = self.forward(x_valid)
                sns.c = cost(sns.Tvalid, sns.pYvalid)
                sns.costs.append(sns.c)
                sns.e = error_rate(y_valid, argmax(sns.pYvalid, axis=1))
                print("i:", i, "cost:", sns.c, "error:", sns.e)
                if sns.e < sns.best_validation_error:
                    sns.best_validation_error = sns.e
        print("best_validation_error:", sns.best_validation_error)
        if sns.show_fig:
            plot(sns.costs)

    def forward(self, x_data):
        """
        forward method
        """
        return softmax(x_data.dot(self.w_data) + self.b_data)

    def predict(self, x_data):
        """
        predict
        """
        py_data = self.forward(x_data)
        return argmax(py_data, axis=1)

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
    x_train, y_train, x_valid, y_valid = get_data()
    model = LogisticModel()
    model.fit(x_train, y_train, x_valid, y_valid)
    print(model.score(x_valid, y_valid))


if __name__ == "__main__":
    main()
    show()
