"""
Bayesin Machine Learning in Python: A/B Testing
"""
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


class Bandit:
    """
    define bandit, there's no "pull arm" here
    since that's technically now the user/client
    """

    def __init__(self, name):
        self.clks = 0
        self.views = 0
        self.name = name

    def sample(self):
        """
        Beta(1, 1) is the prior
        """
        a_var = 1 + self.clks
        b_var = 1 + self.views - self.clks
        return np.random.beta(a_var, b_var)

    def add_click(self):
        """
        add clicks function
        """
        self.clks += 1

    def add_view(self):
        """
        add view function
        """
        self.views += 1
        if self.views % 50 == 0:
            print("%s: clks=%s, views=%s" % (self.name, self.clks, self.views))


banditA = Bandit("A")
banditB = Bandit("B")


@app.route("/get_ad")
def get_ad():
    """
    get ad function
    """
    if banditA.sample() > banditB.sample():
        ad_var = "A"
        banditA.add_view()
    else:
        ad_var = "B"
        banditB.add_view()
    return jsonify({"advertisement_id": ad_var})


@app.route("/click_ad", methods=["POST"])
def click_ad():
    """
    click ad function
    """
    result = "OK"
    if request.form["advertisement_id"] == "A":
        banditA.add_click()
    elif request.form["advertisement_id"] == "B":
        banditB.add_click()
    else:
        result = "Invalid Input."
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="8888")
