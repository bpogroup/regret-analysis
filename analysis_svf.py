import random
from math import sqrt
import scipy.stats as st
from statistics import mean, variance
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def generate_jobs_gamma(p_means, p_sigma2s):
    jobs = []
    for i in range(len(p_means)):
        p_mean = p_means[i]
        p_sigma2 = p_sigma2s[i]

        k = 1 / 4  # keep fixed to keep the skewness fixed
        theta = sqrt(p_sigma2 / k)  # theta = sqrt(sigma^2/k)
        m = k * theta  # mean = k * theta
        error = scipy.stats.gamma.rvs(k, scale=theta) - m  # deduct the mean to keep the error around 0

        y = p_mean + error

        jobs.append((p_mean, p_sigma2, y))

    return jobs


def cost_svf(tau, jobs):
    result = 0
    arrival_time = 0  # arrival time of current job
    start_time = 0  # start time of current job
    completion_time = 0  # completion time of previous job
    for i in range(len(tau)):
        job = jobs[tau[i]]

        waiting_time = max(0, start_time - arrival_time)
        idle_time = max(0, start_time - completion_time)

        result += waiting_time + idle_time  # waiting time and idle time weigh equally

        arrival_time += job[0]
        completion_time = start_time + job[2]
        start_time = max(arrival_time, completion_time)
    return result


def schedule_svf(jobs):
    index_x_job = [(i, jobs[i]) for i in range(len(jobs))]
    index_x_job.sort(key=lambda x: x[1][1])
    return [ixj[0] for ixj in index_x_job]


def regret(jobs):
    # If we schedule the jobs according to the actual times, we always have 0 cost
    # because clients arrive at the exact moment they can start being helped.
    # So the regret equals the costs of the actual arrival times.

    schedule_predicted = schedule_svf(jobs)
    costs_actual = cost_svf(schedule_predicted, jobs)

    return costs_actual


def estimate_regret(p_means, p_sigma2s, nr_experiments):
    regrets = []
    for _ in range(nr_experiments):
        jobs = generate_jobs_gamma(p_means, p_sigma2s)

        regrets.append(regret(jobs))

    avg = mean(regrets)
    ci = st.t.interval(0.95, len(regrets) - 1, loc=avg, scale=st.sem(regrets))

    return avg, avg - ci[0]


def line_with_ci(series, series2, x_lab, y_lab, y2_lab, fig_file=None):
    fig, ax1 = plt.subplots()

    x = series.keys()
    y = [m for (m, h) in series.values()]
    ci_bottom = [m-h for (m, h) in series.values()]
    ci_top = [m+h for (m, h) in series.values()]

    ax1.plot(x, y)
    ax1.fill_between(x, ci_bottom, ci_top, color='blue', alpha=0.1)
    ax1.set_xlabel(x_lab)
    ax1.set_ylabel(y_lab)

    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_lab, color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax2.plot(x, series2.values(), color="red", linestyle=":")

    fig.tight_layout()
    if fig_file is not None:
        plt.savefig(fig_file, format="pdf")
    plt.show()


def estimate_mse(means, sigma2s):
    mse_est = 0
    n = 10000
    for i in range(n):
        jobs = generate_jobs_gamma(means, sigma2s)
        mse = 0
        for job in jobs:
            mse += (job[2] - job[0])**2
        mse /= len(jobs)
        mse_est += mse
    return mse_est/n


def experiment():
    sigma2_x_regret = dict()
    mse_x_regret = dict()
    for sigma2 in [s/10 for s in range(1, 10)] + [s/10 for s in range(10, 50, 5)] + [s/10 for s in range(50, 110, 10)]:
        means = [1, 1]
        sigma2s = [sigma2, 20-sigma2]

        rgt = estimate_regret(means, sigma2s, 10000)
        mse = sum(sigma2s)/len(sigma2s)
        spread = 20-2*sigma2
        print(spread, mse, rgt)

        sigma2_x_regret[spread] = rgt
        mse_x_regret[spread] = mse

    line_with_ci(sigma2_x_regret, mse_x_regret, r'spread $\sigma^2$', 'regret', 'mse')


experiment()
