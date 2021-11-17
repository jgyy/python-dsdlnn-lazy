"""
ann tf script
"""
from types import SimpleNamespace
from numpy import float32
from tensorflow import Variable, compat, matmul, argmax, reduce_mean
from matplotlib.pyplot import plot, show
from util import get_data, y2indicator, error_rate, init_weight_and_bias
from sklearn.utils import shuffle


class HiddenLayer:
    """
    hidden layer class
    """

    def __init__(self, M1, M2, an_id):
        self.an_id = an_id
        self.m_1 = M1
        self.m_2 = M2
        w_0, b_0 = init_weight_and_bias(M1, M2)
        self.w_0 = Variable(w_0.astype(float32))
        self.b_0 = Variable(b_0.astype(float32))
        self.params = [self.w_0, self.b_0]

    def __call__(self):
        print(self)

    def forward(self, x_0):
        """
        forward method
        """
        return compat.v1.nn.relu(matmul(x_0, self.w_0) + self.b_0)


class ANN:
    """
    ann class
    """

    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layers = []
        self.w_0 = None
        self.b_0 = None
        self.params = []

    def fit(self, x_0, y_0, x_valid, y_valid):
        """
        ann fit method
        """
        sns = SimpleNamespace()
        sns.learning_rate, sns.mu, sns.decay, sns.reg = 1e-2, 0.99, 0.999, 1e-3
        sns.epochs, sns.batch_sz, sns.show_fig = 10, 100, True
        sns.K = len(set(y_0))
        x_0, y_0 = shuffle(x_0, y_0)
        x_0 = x_0.astype(float32)
        y_0 = y2indicator(y_0).astype(float32)
        sns.Yvalid_flat = y_valid
        y_valid = y2indicator(y_valid).astype(float32)
        sns.N, sns.D = x_0.shape
        self.hidden_layers = []
        sns.M1 = sns.D
        sns.count = 0
        for m_2 in self.hidden_layer_sizes:
            print(sns.M1, m_2, sns.count)
            sns.h = HiddenLayer(sns.M1, m_2, sns.count)
            self.hidden_layers.append(sns.h)
            sns.M1 = m_2
            sns.count += 1
        sns.W, sns.b = init_weight_and_bias(sns.M1, sns.K)
        self.w_0 = Variable(sns.W.astype(float32))
        self.b_0 = Variable(sns.b.astype(float32))
        self.params = [self.w_0, self.b_0]
        for h_0 in self.hidden_layers:
            self.params += h_0.params
        sns.tfX = compat.v1.placeholder(float32, shape=(None, sns.D), name="X")
        sns.tfT = compat.v1.placeholder(float32, shape=(None, sns.K), name="T")
        sns.act = self.forward(sns.tfX)
        sns.rcost = sns.reg * sum([compat.v1.nn.l2_loss(p) for p in self.params])
        sns.cost = (
            reduce_mean(
                compat.v1.nn.softmax_cross_entropy_with_logits(None, sns.tfT, sns.act)
            )
            + sns.rcost
        )
        sns.prediction = self.predict(sns.tfX)
        sns.train_op = compat.v1.train.RMSPropOptimizer(
            sns.learning_rate, decay=sns.decay, momentum=sns.mu
        ).minimize(sns.cost)
        sns.n_batches = sns.N // sns.batch_sz
        sns.costs = []
        sns.init = compat.v1.global_variables_initializer()
        with compat.v1.Session() as session:
            session.run(sns.init)
            for i in range(sns.epochs):
                x_0, y_0 = shuffle(x_0, y_0)
                for j in range(sns.n_batches):
                    sns.Xbatch = x_0[
                        j * sns.batch_sz : (j * sns.batch_sz + sns.batch_sz)
                    ]
                    sns.Ybatch = y_0[
                        j * sns.batch_sz : (j * sns.batch_sz + sns.batch_sz)
                    ]
                    session.run(
                        sns.train_op,
                        feed_dict={sns.tfX: sns.Xbatch, sns.tfT: sns.Ybatch},
                    )
                    if j % 20 == 0:
                        sns.c = session.run(
                            sns.cost, feed_dict={sns.tfX: x_valid, sns.tfT: y_valid}
                        )
                        sns.costs.append(sns.c)
                        sns.p = session.run(
                            sns.prediction,
                            feed_dict={sns.tfX: x_valid, sns.tfT: y_valid},
                        )
                        sns.e = error_rate(sns.Yvalid_flat, sns.p)
                        print(
                            "i:",
                            i,
                            "j:",
                            j,
                            "nb:",
                            sns.n_batches,
                            "cost:",
                            sns.c,
                            "error rate:",
                            sns.e,
                        )
        if sns.show_fig:
            plot(sns.costs)
            show()

    def forward(self, x_0):
        """
        ann forward method
        """
        z_0 = x_0
        for h_0 in self.hidden_layers:
            z_0 = h_0.forward(z_0)
        return matmul(z_0, self.w_0) + self.b_0

    def predict(self, x_0):
        """
        ann predict method
        """
        act = self.forward(x_0)
        return argmax(act, 1)


def main():
    """
    main function
    """
    x_train, y_train, x_valid, y_valid = get_data()
    model = ANN([2000, 1000, 500])
    model.fit(x_train, y_train, x_valid, y_valid)


if __name__ == "__main__":
    compat.v1.disable_eager_execution()
    main()
