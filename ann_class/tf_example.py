"""
Neural network in TensorFlow very simple example.
"""
from numpy import vstack, array, zeros, mean, float32 as nfloat32
from numpy.random import randn
from matplotlib.pyplot import scatter, show
from tensorflow import (
    Variable,
    compat,
    sigmoid,
    matmul,
    reduce_mean,
    argmax,
    float32 as tfloat32,
)


def init_weights(shape):
    """
    tensor flow variables are not the same as regular Python variables
    """
    return Variable(compat.v1.random_normal(shape, stddev=0.01))


def forward(x_0, w_1, b_1, w_2, b_2):
    """
    forward function
    """
    z_var = sigmoid(matmul(x_0, w_1) + b_1)
    return matmul(z_var, w_2) + b_2


def main():
    """
    main function
    """
    dic = {"Nclass": 500, "D": 2, "M": 3, "K": 3}
    dic |= {
        "X1": randn(dic["Nclass"], dic["D"]) + array([0, -2]),
        "X2": randn(dic["Nclass"], dic["D"]) + array([2, 2]),
        "X3": randn(dic["Nclass"], dic["D"]) + array([-2, 2]),
    }
    dic |= {
        "X": vstack([dic["X1"], dic["X2"], dic["X3"]]).astype(nfloat32),
        "Y": array([0] * dic["Nclass"] + [1] * dic["Nclass"] + [2] * dic["Nclass"]),
    }
    scatter(dic["X"][:, 0], dic["X"][:, 1], c=dic["Y"], s=100, alpha=0.5)
    dic |= {"N": len(dic["Y"])}
    dic |= {"T": zeros((dic["N"], dic["K"]))}
    for i in range(dic["N"]):
        dic["T"][i, dic["Y"][i]] = 1
    compat.v1.disable_eager_execution()
    tf_x = compat.v1.placeholder(dtype=tfloat32, shape=[None, dic["D"]])
    tf_y = compat.v1.placeholder(dtype=tfloat32, shape=[None, dic["K"]])
    w_1 = init_weights([dic["D"], dic["M"]])
    b_1 = init_weights([dic["M"]])
    w_2 = init_weights([dic["M"], dic["K"]])
    b_2 = init_weights([dic["K"]])
    logits = forward(tf_x, w_1, b_1, w_2, b_2)
    cost = reduce_mean(
        compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=logits)
    )
    train_op = compat.v1.train.GradientDescentOptimizer(0.05).minimize(cost)
    predict_op = argmax(logits, 1)
    sess = compat.v1.Session()
    init = compat.v1.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train_op, feed_dict={tf_x: dic["X"], tf_y: dic["T"]})
        pred = sess.run(predict_op, feed_dict={tf_x: dic["X"], tf_y: dic["T"]})
        if i % 100 == 0:
            print("Accuracy:", mean(dic["Y"] == pred))


if __name__ == "__main__":
    main()
    show()
