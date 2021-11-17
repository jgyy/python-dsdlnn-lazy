"""
batch norm tensorflow script
"""
from types import SimpleNamespace
from numpy import sqrt, float32, int32, ones, zeros, argmax, mean
from numpy.random import randn
from matplotlib.pyplot import plot, show
from tensorflow import Variable, compat, matmul, control_dependencies, reduce_mean
from sklearn.utils import shuffle
from util import get_normalized_data


def init_weight(m_1, m_2):
    """
    init weight function
    """
    return randn(m_1, m_2) * sqrt(2.0 / m_1)


class HiddenLayerBatchNorm:
    """
    hidden layer batch norm class
    """

    def __init__(self, M1, M2, f):
        self.f_0 = f
        w_0 = init_weight(M1, M2).astype(float32)
        gamma = ones(M2).astype(float32)
        beta = zeros(M2).astype(float32)
        self.w_0 = Variable(w_0)
        self.gamma = Variable(gamma)
        self.beta = Variable(beta)
        self.running_mean = Variable(zeros(M2).astype(float32), trainable=False)
        self.running_var = Variable(zeros(M2).astype(float32), trainable=False)

    def __call__(self):
        print(self)

    def forward(self, x_0, is_training, decay=0.9):
        """
        forward method
        """
        activation = matmul(x_0, self.w_0)
        if is_training:
            batch_mean, batch_var = compat.v1.nn.moments(activation, [0])
            update_running_mean = compat.v1.assign(
                self.running_mean, self.running_mean * decay + batch_mean * (1 - decay)
            )
            update_running_var = compat.v1.assign(
                self.running_var, self.running_var * decay + batch_var * (1 - decay)
            )
            with control_dependencies([update_running_mean, update_running_var]):
                out = compat.v1.nn.batch_normalization(
                    activation, batch_mean, batch_var, self.beta, self.gamma, 1e-4
                )
        else:
            out = compat.v1.nn.batch_normalization(
                activation,
                self.running_mean,
                self.running_var,
                self.beta,
                self.gamma,
                1e-4,
            )
        return self.f_0(out)


class HiddenLayer:
    """
    hidden layer class
    """

    def __init__(self, M1, M2, f):
        self.f_0 = f
        w_0 = randn(M1, M2) * sqrt(2.0 / M1)
        b_0 = zeros(M2)
        self.w_0 = Variable(w_0.astype(float32))
        self.b_0 = Variable(b_0.astype(float32))

    def __call__(self):
        print(self)

    def forward(self, x_0):
        """
        forward method
        """
        return self.f_0(matmul(x_0, self.w_0) + self.b_0)


class ANN:
    """
    ann class
    """

    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.session = None
        self.tf_x = None
        self.predict_op = None
        self.layers = []
        compat.v1.disable_eager_execution()

    def set_session(self, session):
        """
        set session method
        """
        self.session = session

    def fit(self, x_train, y_train, x_test, y_test):
        """
        fit method
        """
        sns = SimpleNamespace()
        sns.show_fig = True
        sns.learning_rate = 1e-2
        sns.epochs = 15
        sns.batch_sz = 100
        sns.print_period = 100
        sns.activation = compat.v1.nn.relu
        x_train = x_train.astype(float32)
        y_train = y_train.astype(int32)
        sns.N, sns.D = x_train.shape
        self.layers = []
        sns.M1 = sns.D
        for m_2 in self.hidden_layer_sizes:
            sns.h = HiddenLayerBatchNorm(sns.M1, m_2, sns.activation)
            self.layers.append(sns.h)
            sns.M1 = m_2
        sns.K = len(set(y_train))
        sns.h = HiddenLayer(sns.M1, sns.K, lambda x: x)
        self.layers.append(sns.h)
        if sns.batch_sz is None:
            sns.batch_sz = sns.N
        sns.tfX = compat.v1.placeholder(float32, shape=(None, sns.D), name="X")
        sns.tfY = compat.v1.placeholder(int32, shape=(None,), name="Y")
        self.tf_x = sns.tfX
        sns.logits = self.forward(sns.tfX, is_training=True)
        sns.cost = reduce_mean(
            compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
                None, sns.tfY, sns.logits
            )
        )
        sns.train_op = compat.v1.train.MomentumOptimizer(
            sns.learning_rate, momentum=0.9, use_nesterov=True
        ).minimize(sns.cost)
        sns.test_logits = self.forward(sns.tfX, is_training=False)
        self.predict_op = compat.v1.argmax(sns.test_logits, 1)
        self.session.run(compat.v1.global_variables_initializer())
        sns.n_batches = sns.N // sns.batch_sz
        sns.costs = []
        for i in range(sns.epochs):
            if sns.n_batches > 1:
                x_train, y_train = shuffle(x_train, y_train)
            for j in range(sns.n_batches):
                Xbatch = x_train[j * sns.batch_sz : (j * sns.batch_sz + sns.batch_sz)]
                Ybatch = y_train[j * sns.batch_sz : (j * sns.batch_sz + sns.batch_sz)]
                c, _, lgts = self.session.run(
                    [sns.cost, sns.train_op, sns.logits],
                    feed_dict={sns.tfX: Xbatch, sns.tfY: Ybatch},
                )
                sns.costs.append(c)
                if (j + 1) % sns.print_period == 0:
                    acc = mean(Ybatch == argmax(lgts, axis=1))
                    print(
                        "epoch:",
                        i,
                        "batch:",
                        j,
                        "n_batches:",
                        sns.n_batches,
                        "cost:",
                        c,
                        "acc: %.2f" % acc,
                    )
            print(
                "Train acc:",
                self.score(x_train, y_train),
                "Test acc:",
                self.score(x_test, y_test),
            )
        if sns.show_fig:
            plot(sns.costs)

    def forward(self, x_0, is_training):
        """
        forward method
        """
        out = x_0
        for h_0 in self.layers[:-1]:
            out = h_0.forward(out, is_training)
        out = self.layers[-1].forward(out)
        return out

    def score(self, x_0, y_0):
        """
        score method
        """
        p_0 = self.predict(x_0)
        return mean(y_0 == p_0)

    def predict(self, x_0):
        """
        predict method
        """
        return self.session.run(self.predict_op, feed_dict={self.tf_x: x_0})


def main():
    """
    main function
    """
    x_train, x_test, y_train, y_test = get_normalized_data()
    ann = ANN([500, 300])
    session = compat.v1.InteractiveSession()
    ann.set_session(session)
    ann.fit(x_train, y_train, x_test, y_test)
    print("Train accuracy:", ann.score(x_train, y_train))
    print("Test accuracy:", ann.score(x_test, y_test))


if __name__ == "__main__":
    main()
    show()
