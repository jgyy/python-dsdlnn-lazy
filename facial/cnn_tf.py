"""
cnn tf script
"""
from types import SimpleNamespace
from numpy import sqrt, float32, prod, zeros, argmax as nargmax
from numpy.random import randn
from tensorflow import (
    Variable,
    nn,
    compat,
    reduce_mean,
    reshape,
    matmul,
    argmax as targmax,
)
from matplotlib.pyplot import plot, show
from sklearn.utils import shuffle
from util import get_image_data, error_rate, init_weight_and_bias, y2indicator
from ann_tf import HiddenLayer


def init_filter(shape, poolsz):
    """
    init filter function
    """
    w_0 = (
        randn(*shape)
        * sqrt(2)
        / sqrt(prod(shape[:-1]) + shape[-1] * prod(shape[:-2] / prod(poolsz)))
    )
    return w_0.astype(float32)


class ConvPoolLayer:
    """
    conv pool layer class
    """

    def __init__(self, mi, mo, fw=5, fh=5):
        poolsz = (2, 2)
        s_z = (fw, fh, mi, mo)
        w_0 = init_filter(s_z, poolsz)
        self.w_0 = Variable(w_0)
        b_0 = zeros(mo, dtype=float32)
        self.b_0 = Variable(b_0)
        self.poolsz = poolsz
        self.params = [self.w_0, self.b_0]

    def __call__(self):
        print(self)

    def forward(self, x_0):
        """
        forward method
        """
        conv_out = nn.conv2d(x_0, self.w_0, strides=[1, 1, 1, 1], padding="SAME")
        conv_out = nn.bias_add(conv_out, self.b_0)
        p_1, p_2 = self.poolsz
        pool_out = nn.max_pool(
            conv_out, ksize=[1, p_1, p_2, 1], strides=[1, p_1, p_2, 1], padding="SAME"
        )
        return nn.relu(pool_out)


class CNN:
    """
    cnn class
    """

    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.convpool_layers = []
        self.hidden_layers = []
        self.w_0 = None
        self.b_0 = None
        self.params = []

    def fit(self, x_0, y_0, x_valid, y_valid):
        """
        fit method
        """
        sns = SimpleNamespace(
            lr=1e-2,
            mu=0.9,
            reg=1e-3,
            decay=0.99999,
            eps=1e-10,
            batch_sz=30,
            epochs=5,
            show_fig=True,
        )
        sns.lr = float32(sns.lr)
        sns.mu = float32(sns.mu)
        sns.reg = float32(sns.reg)
        sns.decay = float32(sns.decay)
        sns.eps = float32(sns.eps)
        sns.K = len(set(y_0))
        sns.X, sns.Y = shuffle(x_0, y_0)
        sns.X = sns.X.astype(float32)
        sns.Y = y2indicator(sns.Y).astype(float32)
        sns.Xvalid = x_valid
        sns.Yvalid = y2indicator(y_valid).astype(float32)
        sns.Yvalid_flat = nargmax(sns.Yvalid, axis=1)
        sns.N, sns.width, sns.height, sns.c = sns.X.shape
        sns.mi = sns.c
        sns.outw = sns.width
        sns.outh = sns.height
        self.convpool_layers = []
        for m_o, f_w, f_h in self.convpool_layer_sizes:
            sns.layer = ConvPoolLayer(sns.mi, m_o, f_w, f_h)
            self.convpool_layers.append(sns.layer)
            sns.outw = sns.outw // 2
            sns.outh = sns.outh // 2
            sns.mi = m_o
        self.hidden_layers = []
        sns.M1 = self.convpool_layer_sizes[-1][0] * sns.outw * sns.outh
        sns.count = 0
        for m_2 in self.hidden_layer_sizes:
            sns.h = HiddenLayer(sns.M1, m_2, sns.count)
            self.hidden_layers.append(sns.h)
            sns.M1 = m_2
            sns.count += 1
        sns.W, sns.b = init_weight_and_bias(sns.M1, sns.K)
        self.w_0 = Variable(sns.W, "W_logreg")
        self.b_0 = Variable(sns.b, "b_logreg")
        self.params = [self.w_0, self.b_0]
        for hidden in self.convpool_layers:
            self.params += hidden.params
        for hidden in self.hidden_layers:
            self.params += hidden.params
        sns.tfX = compat.v1.placeholder(
            float32, shape=(None, sns.width, sns.height, sns.c), name="X"
        )
        sns.tfY = compat.v1.placeholder(float32, shape=(None, sns.K), name="Y")
        sns.act = self.forward(sns.tfX)
        sns.rcost = sns.reg * sum([nn.l2_loss(p) for p in self.params])
        sns.cost = (
            reduce_mean(
                nn.softmax_cross_entropy_with_logits(logits=sns.act, labels=sns.tfY)
            )
            + sns.rcost
        )
        sns.prediction = self.predict(sns.tfX)
        sns.train_op = compat.v1.train.RMSPropOptimizer(
            sns.lr, decay=sns.decay, momentum=sns.mu
        ).minimize(sns.cost)
        sns.n_batches = sns.N // sns.batch_sz
        sns.costs = []
        sns.init = compat.v1.global_variables_initializer()
        self._session(sns)

    @staticmethod
    def _session(sns):
        with compat.v1.Session() as session:
            session.run(sns.init)
            for i in range(sns.epochs):
                sns.X, sns.Y = shuffle(sns.X, sns.Y)
                for j in range(sns.n_batches):
                    sns.Xbatch = sns.X[
                        j * sns.batch_sz : (j * sns.batch_sz + sns.batch_sz)
                    ]
                    sns.Ybatch = sns.Y[
                        j * sns.batch_sz : (j * sns.batch_sz + sns.batch_sz)
                    ]
                    session.run(
                        sns.train_op,
                        feed_dict={sns.tfX: sns.Xbatch, sns.tfY: sns.Ybatch},
                    )
                    if j % 20 == 0:
                        sns.c = session.run(
                            sns.cost,
                            feed_dict={sns.tfX: sns.Xvalid, sns.tfY: sns.Yvalid},
                        )
                        sns.costs.append(sns.c)
                        sns.p = session.run(
                            sns.prediction,
                            feed_dict={sns.tfX: sns.Xvalid, sns.tfY: sns.Yvalid},
                        )
                        sns.e = error_rate(sns.Yvalid_flat, sns.p)
                        print(
                            f"i: {i} j: {j} nb: {sns.n_batches} cost: {sns.c} error rate: {sns.e}"
                        )
        if sns.show_fig:
            plot(sns.costs)

    def forward(self, x_0):
        """
        forward method
        """
        z_0 = x_0
        for c_0 in self.convpool_layers:
            z_0 = c_0.forward(z_0)
        z_shape = z_0.get_shape().as_list()
        z_0 = reshape(z_0, [-1, prod(z_shape[1:])])
        for h_0 in self.hidden_layers:
            z_0 = h_0.forward(z_0)
        return matmul(z_0, self.w_0) + self.b_0

    def predict(self, x_0):
        """
        predict method
        """
        p_y = self.forward(x_0)
        return targmax(p_y, 1)


def main():
    """
    main function
    """
    x_train, y_train, x_valid, y_valid = get_image_data()
    x_train = x_train.transpose((0, 2, 3, 1))
    x_valid = x_valid.transpose((0, 2, 3, 1))
    model = CNN(
        convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
        hidden_layer_sizes=[500, 300],
    )
    model.fit(x_train, y_train, x_valid, y_valid)


if __name__ == "__main__":
    compat.v1.disable_eager_execution()
    main()
    show()
