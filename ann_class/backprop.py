"""
backpropagation example for deep learning in python class with sigmoid activation
"""
from numpy import argmax, array, vstack, zeros, exp, log
from numpy.random import randn
from matplotlib.pyplot import show, plot, scatter, figure


def forward(x_var, w1_var, b1_var, w2_var, b2_var):
    """
    forward function
    """
    z_var = 1 / (1 + exp(-x_var.dot(w1_var) - b1_var))
    a_var = z_var.dot(w2_var) + b2_var
    exp_a = exp(a_var)
    y_var = exp_a / exp_a.sum(axis=1, keepdims=True)
    return y_var, z_var


def classification_rate(y_var, p_var):
    """
    determine the classification rate (num correct / num total)
    """
    n_correct = 0
    n_total = 0
    for index, value in enumerate(y_var):
        n_total += 1
        if value == p_var[index]:
            n_correct += 1
    return float(n_correct) / n_total


def derivative_w2(z_var, t_var, y_var):
    """
    N, K = T.shape
    M = Z.shape[1]
    ret1 = np.zeros((M, K))
    for n in xrange(N):
        for m in xrange(M):
            for k in xrange(K):
                ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]
    """
    ret4 = z_var.T.dot(t_var - y_var)
    return ret4


def derivative_w1(x_var, z_var, t_var, y_var, w2_var):
    """
    N, D = X.shape
    M, K = W2.shape
    ret1 = np.zeros((X.shape[1], M))
    for n in xrange(N):
        for k in xrange(K):
            for m in xrange(M):
                for d in xrange(D):
                    ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1 - Z[n,m])*X[n,d]
    """
    dz_var = (t_var - y_var).dot(w2_var.T) * z_var * (1 - z_var)
    ret2 = x_var.T.dot(dz_var)
    return ret2


def derivative_b2(t_var, y_var):
    """
    derivative b2 function
    """
    return (t_var - y_var).sum(axis=0)


def derivative_b1(t_var, y_var, w2_var, z_var):
    """
    derivative b1 function
    """
    return ((t_var - y_var).dot(w2_var.T) * z_var * (1 - z_var)).sum(axis=0)


def cost(t_var, y_var):
    """
    cost function
    """
    tot = t_var * log(y_var)
    return tot.sum()


def main():
    """
    create the data
    """
    var = {"Nclass": 500, "D": 2, "M": 3, "K": 3}
    var |= {
        "X1": randn(var["Nclass"], var["D"]) + array([0, -2]),
        "X2": randn(var["Nclass"], var["D"]) + array([2, 2]),
        "X3": randn(var["Nclass"], var["D"]) + array([-2, 2]),
    }
    var |= {
        "X": vstack([var["X1"], var["X2"], var["X3"]]),
        "Y": array([0] * var["Nclass"] + [1] * var["Nclass"] + [2] * var["Nclass"]),
    }
    var |= {"N": len(var["Y"])}
    var |= {"T": zeros((var["N"], var["K"]))}
    for i in range(var["N"]):
        var["T"][i, var["Y"][i]] = 1

    figure()
    scatter(var["X"][:, 0], var["X"][:, 1], c=var["Y"], s=100, alpha=0.5)
    var |= {
        "W1": randn(var["D"], var["M"]),
        "b1": randn(var["M"]),
        "W2": randn(var["M"], var["K"]),
        "b2": randn(var["K"])
    }
    learning_rate = 1e-3
    costs = []
    for epoch in range(1000):
        output, hidden = forward(var["X"], var["W1"], var["b1"], var["W2"], var["b2"])
        if epoch % 100 == 0:
            c_var = cost(var["T"], output)
            p_var = argmax(output, axis=1)
            r_var = classification_rate(var["Y"], p_var)
            print("cost:", c_var, "classification_rate:", r_var)
            costs.append(c_var)

        gw2 = derivative_w2(hidden, var["T"], output)
        gb2 = derivative_b2(var["T"], output)
        gw1 = derivative_w1(var["X"], hidden, var["T"], output, var["W2"])
        gb1 = derivative_b1(var["T"], output, var["W2"], hidden)

        var["W2"] += learning_rate * gw2
        var["b2"] += learning_rate * gb2
        var["W1"] += learning_rate * gw1
        var["b1"] += learning_rate * gb1

    figure()
    plot(costs)


if __name__ == "__main__":
    main()
    show()
