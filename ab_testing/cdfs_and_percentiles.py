"""
cdfs and percentiles script
"""
from scipy.stats import norm


def main():
    """
    main function
    """
    mu_data = 170
    sd_data = 7
    x_data = norm.rvs(loc=mu_data, scale=sd_data, size=100)

    print("maximum likelihood mean", x_data.mean())
    print("maximum likelihood variance", x_data.var())
    print("maximum likelihood std", x_data.std())
    print("unbiased variance", x_data.var(ddof=1))
    print("unbiased std", x_data.std(ddof=1))
    print(
        "at what height are you in the 95th percentile?",
        norm.ppf(0.95, loc=mu_data, scale=sd_data),
    )
    print(
        "you are 160 cm tall, what percentile are you in?",
        norm.cdf(160, loc=mu_data, scale=sd_data),
    )
    print(
        "you are 180 cm tall, what is the probability that someone is taller than you?",
        1 - norm.cdf(180, loc=mu_data, scale=sd_data),
    )


if __name__ == "__main__":
    main()
