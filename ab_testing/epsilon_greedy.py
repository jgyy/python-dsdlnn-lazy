"""
Bayesin Machine Learning in Python: A/B Testing
"""
from matplotlib.pyplot import plot, show
from numpy import max as nmax, argwhere, amax, zeros, argmax, cumsum, arange, ones
from numpy.random import choice, random, randint

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class BanditArm:
    """
    bandit arm class
    """

    def __init__(self, p):
        self.p_var = p
        self.p_estimate = 0.0
        self.n_var = 0.0

    def pull(self):
        """
        pull method
        """
        return random() < self.p_var

    def update(self, x_var):
        """
        update method
        """
        self.n_var += 1.0
        self.p_estimate = ((self.n_var - 1) * self.p_estimate + x_var) / self.n_var


def choose_random_argmax(a_var):
    """
    choose random argmax function
    """
    idx = argwhere(amax(a_var) == a_var).flatten()
    return choice(idx)


def experiment():
    """
    experiment function
    """
    bandits = [BanditArm(p) for p in BANDIT_PROBABILITIES]
    rewards = zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = argmax([b.p_var for b in bandits])
    print("optimal j:", optimal_j)
    for i in range(NUM_TRIALS):
        if random() < EPS:
            num_times_explored += 1
            j = randint(len(bandits))
        else:
            num_times_exploited += 1
            j = choose_random_argmax([b.p_estimate for b in bandits])
        if j == optimal_j:
            num_optimal += 1
        x_var = bandits[j].pull()
        rewards[i] = x_var
        bandits[j].update(x_var)
    for b_var in bandits:
        print("mean estimate:", b_var.p_estimate)
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)
    cumulative_rewards = cumsum(rewards)
    win_rates = cumulative_rewards / (arange(NUM_TRIALS) + 1)
    plot(win_rates)
    plot(ones(NUM_TRIALS) * nmax(BANDIT_PROBABILITIES))


if __name__ == "__main__":
    experiment()
    show()
