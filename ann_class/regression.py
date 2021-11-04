"""
regression script
"""
from numpy import sqrt, zeros, outer, linspace, meshgrid, vstack, abs as nabs
from numpy.random import random, randn
from matplotlib.pyplot import show, figure, plot, scatter


def forward(x_var, w_var, b_var, v_var, c_var):
    """
    how to get the output, consider the params global
    """
    z_var = x_var.dot(w_var) + b_var
    z_var = z_var * (z_var > 0)
    y_hat = z_var.dot(v_var) + c_var
    return z_var, y_hat


def derivative_v(z_var, y_var, y_hat):
    """
    how to train the params
    """
    return (y_var - y_hat).dot(z_var)


def derivative_c(y_var, y_hat):
    """
    derivative c function
    """
    return (y_var - y_hat).sum()


def derivative_w(x_var, z_var, y_var, y_hat, v_var):
    """
    this is for tanh activation
    """
    d_z = outer(y_var - y_hat, v_var) * (z_var > 0)
    return x_var.T.dot(d_z)


def derivative_b(z_var, y_var, y_hat, v_hat):
    """
    this is for relu activation
    """
    d_z = outer(y_var - y_hat, v_hat) * (z_var > 0)
    return d_z.sum(axis=0)


def update(dic, z_var, y_hat, learning_rate=1e-4):
    """
    update function
    X, Z, Y, Yhat, W, b, V, c, learning_rate=1e-4
    """
    g_v = derivative_v(z_var, dic["Y"], y_hat)
    g_c = derivative_c(dic["Y"], y_hat)
    g_w = derivative_w(dic["X"], z_var, dic["Y"], y_hat, dic["V"])
    g_b = derivative_b(z_var, dic["Y"], y_hat, dic["V"])

    dic["V"] += learning_rate * g_v
    dic["c"] += learning_rate * g_c
    dic["W"] += learning_rate * g_w
    dic["b"] += learning_rate * g_b

    return dic["W"], dic["b"], dic["V"], dic["c"]


def get_cost(y_var, y_hat):
    """
    so we can plot the costs later
    """
    return ((y_var - y_hat) ** 2).mean()


def main():
    """
    main function
    """
    dic = {"N": 500}
    dic |= {"X": random((dic["N"], 2)) * 4 - 2}
    dic |= {"Y": dic["X"][:, 0] * dic["X"][:, 1]}
    fig = figure()
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(dic["X"][:, 0], dic["X"][:, 1], dic["Y"])
    dic |= {"D": 2, "M": 100}
    dic |= {
        "W": randn(dic["D"], dic["M"]) / sqrt(dic["D"]),
        "b": zeros(dic["M"]),
        "V": randn(dic["M"]) / sqrt(dic["M"]),
        "c": 0,
    }
    costs = []
    for i in range(200):
        z_var, y_hat = forward(dic["X"], dic["W"], dic["b"], dic["V"], dic["c"])
        dic["W"], dic["b"], dic["V"], dic["c"] = update(dic, z_var, y_hat)
        cost = get_cost(dic["Y"], y_hat)
        costs.append(cost)
        if i % 25 == 0:
            print(cost)
    figure()
    plot(costs)
    fig = figure()
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(dic["X"][:, 0], dic["X"][:, 1], dic["Y"])
    line = linspace(-2, 2, 20)
    xx_var, yy_var = meshgrid(line, line)
    x_grid = vstack((xx_var.flatten(), yy_var.flatten())).T
    _, y_hat = forward(x_grid, dic["W"], dic["b"], dic["V"], dic["c"])
    axis.plot_trisurf(x_grid[:, 0], x_grid[:, 1], y_hat, linewidth=0.2, antialiased=True)
    y_grid = x_grid[:, 0] * x_grid[:, 1]
    r_var = nabs(y_grid - y_hat)
    figure()
    scatter(x_grid[:, 0], x_grid[:, 1], c=r_var)
    fig = figure()
    axis = fig.add_subplot(111, projection="3d")
    axis.plot_trisurf(x_grid[:, 0], x_grid[:, 1], r_var, linewidth=0.2, antialiased=True)


if __name__ == "__main__":
    main()
    show()
