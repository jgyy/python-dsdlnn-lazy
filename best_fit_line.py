"""
best fit line script
"""
from pulp import LpMinimize, LpStatus, LpProblem, LpVariable, GLPK, value
from numpy import array, linspace
from matplotlib.pyplot import scatter, plot, show


def main():
    """
    main function
    """
    prob = LpProblem("best_fit_line", LpMinimize)
    z_var = LpVariable("z", 0)
    a_var = LpVariable("a", 0)
    c_var = LpVariable("c", 0)
    prob += z_var
    points = [(1, 3), (2, 5), (3, 7), (5, 11), (7, 14), (8, 15), (10, 19)]
    prob += a_var != 0
    for x_var, y_var in points:
        prob += a_var * x_var - y_var - c_var <= z_var
        prob += a_var * x_var - y_var - c_var >= -z_var
    status = prob.solve(GLPK(msg=0))
    print("status:", LpStatus[status])
    print("values:")
    print("\ta:", value(a_var))
    print("\tc:", value(c_var))
    print("\tz:", value(z_var))
    data = array(points)
    scatter(data[:, 0], data[:, 1])
    x_var = linspace(0, 11, 100)
    y_var = value(a_var) * x_var - value(c_var)
    plot(x_var, y_var)


if __name__ == "__main__":
    main()
    show()
