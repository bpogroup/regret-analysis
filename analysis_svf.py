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


def generate_jobs_gamma_f(n, sigma2, f):
    """
    Generates n jobs.
    Returns the list of jobs, with each job is a tuple (x1, x2, y)
    """
    jobs = []
    for i in range(n):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)

        k = 1 / 4  # keep fixed to keep the skewness fixed
        theta = sqrt(sigma2 / k)  # theta = sqrt(sigma^2/k)
        m = k * theta  # mean = k * theta
        error = scipy.stats.gamma.rvs(k, scale=theta) - m  # deduct the mean to keep the error around 0

        y = f(x1, x2) + error

        jobs.append((x1, x2, y))

    return jobs


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


def generate_jobs_normal(p_means, p_sigma2s):
    jobs = []
    for i in range(len(p_means)):
        p_mean = p_means[i]
        p_sigma2 = p_sigma2s[i]

        error = -1
        while error <= 0:
            error = random.normalvariate(0, sqrt(p_sigma2))

        y = p_mean + error

        jobs.append((p_mean, p_sigma2, y))

    return jobs


def generate_jobs_exponential(p_means, p_sigma2s):
    jobs = []
    for i in range(len(p_means)):
        p_mean = p_means[i]
        p_sigma2 = p_sigma2s[i]

        error = np.random.exponential(scale=np.sqrt(p_sigma2)) - np.sqrt(p_sigma2)

        y = p_mean + error

        jobs.append((p_mean, p_sigma2, y))

    return jobs


def generate_jobs_lognormal(p_means, p_sigma2s):
    jobs = []
    for i in range(len(p_means)):
        p_mean = p_means[i]
        p_sigma2 = p_sigma2s[i]

        error = np.random.lognormal(0, sqrt(p_sigma2))

        y = p_mean + error

        jobs.append((p_mean, p_sigma2, y))

    return jobs


def xs(jobs):
    return np.array([[x1, x2] for (x1, x2, y) in jobs])


def ys(jobs):
    return np.array([y for (x1, x2, y) in jobs])


def learn_f_hat(jobs):
    return LinearRegression().fit(xs(jobs), ys(jobs))


def predict_f_hat(model, jobs):
    return model.predict(xs(jobs))


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


def estimate_regret(p_means, p_sigma2s, nr_experiments, generation_function):
    regrets = []
    for _ in range(nr_experiments):
        jobs = generation_function(p_means, p_sigma2s)

        regrets.append(regret(jobs))

    avg = mean(regrets)
    ci = st.t.interval(0.95, len(regrets) - 1, loc=avg, scale=st.sem(regrets))

    return avg, avg - ci[0]


def estimate_regret_learned(p_means, p_sigma2s, mse_s, nr_experiments, generation_function):
    regrets = []
    for _ in range(nr_experiments):
        optimal_jobs = generation_function(p_means, p_sigma2s)
        optimal_schedule = schedule_svf(optimal_jobs)
        optimal_costs = cost_svf(optimal_schedule, optimal_jobs)

        actual_jobs = generation_function(p_means, mse_s)
        actual_schedule = schedule_svf(actual_jobs)
        actual_costs = cost_svf(actual_schedule, actual_jobs)

        regrets.append((actual_costs - optimal_costs)/optimal_costs)

    avg = mean(regrets)
    ci = st.t.interval(0.95, len(regrets) - 1, loc=avg, scale=st.sem(regrets))

    return avg, avg - ci[0]


def line_with_ci(series, x_lab, y_lab, series2=None, y2_lab=None, fig_file=None, x_ticks=None, title=None):
    fontsize = 18
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = series.keys()
    y = [m for (m, h) in series.values()]
    ci_bottom = [m-h for (m, h) in series.values()]
    ci_top = [m+h for (m, h) in series.values()]

    ax1.plot(x, y)
    ax1.fill_between(x, ci_bottom, ci_top, color='blue', alpha=0.1)
    if x_ticks is not None:
        ax1.set_xticks(x_ticks)
    ax1.set_xlabel(x_lab, fontsize=fontsize)
    ax1.set_ylabel(y_lab, fontsize=fontsize)
    if title is not None:
        ax1.set_title(title, fontsize=20)

    if series2 is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(y2_lab, color="red", fontsize=fontsize)
        ax2.tick_params(axis='y', labelcolor="red")
        ax2.plot(x, series2.values(), color="red", linestyle=":")
        for tick in ax2.get_yticklabels():
            tick.set_fontsize(fontsize)

    for tick in ax1.get_xticklabels():
        tick.set_fontsize(fontsize)
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(fontsize)

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


def experiment_vary_mse(nr_jobs, generation_function, filename):
    # different mse, plot regret (keep spread the same to not measure that effect)
    sigma2_x_regret = dict()
    for sigma2 in [1+s/10 for s in range(1, 1000, 10)]:
        means = []
        sigma2s = []
        for i in range(nr_jobs):
            means.append(2)  # the means do not matter for this problem
            sigma2s.append(random.uniform(0, 2 * sigma2))  # we need to vary the MSEs, otherwise it does not make sense

        rgt = estimate_regret(means, sigma2s, 10000, generation_function)
        mse = sum(sigma2s)/len(sigma2s)

        print(mse, rgt)

        sigma2_x_regret[sigma2] = rgt

    line_with_ci(sigma2_x_regret, 'MSE', 'Empirical regret', fig_file=filename, title="Regret vs. MSE")


def single_experiment(generation_function, nr_jobs, nr_learning_samples, f_y, sigma2):
    # different mse, plot regret (keep spread the same to not measure that effect)
    means = []
    sigma2s = []
    mses = []
    for i in range(nr_jobs):
        means.append(2)  # the means do not matter for this problem
        job_sigma2 = (i+1) * (2*sigma2)/nr_jobs  # fixing the sigma's to prevent randomness due to sigma's
        sigma2s.append(job_sigma2)  # we need to vary the MSEs, otherwise it does not make sense

        # get the MSEs
        jobs = generate_jobs_gamma_f(nr_learning_samples, sigma2, f_y)
        model = learn_f_hat(jobs)
        jobs = generate_jobs_gamma_f(10000, sigma2, f_y)
        mse = mean_squared_error(ys(jobs), predict_f_hat(model, jobs))
        mses.append(mse)

    rgt = estimate_regret_learned(means, sigma2s, mses, 10000, generation_function)

    mse = mean(mses)
    mse_ci = st.t.interval(0.95, len(mses) - 1, loc=mse, scale=st.sem(mses))
    mse_ci = mse - mse_ci[0]

    print(sigma2, mse)

    return (mse, mse_ci), rgt


def bar_experiments(experiment_results, x_title, x_labels, fig_file=None):
    fontsize = 16
    bar_width = .4
    rgt_bars = []
    rgt_errs = []
    mse_bars = []
    mse_errs = []
    for mse, rgt in experiment_results:
        rgt_bars.append(rgt[0])
        rgt_errs.append(rgt[1])
        mse_bars.append(mse[0])
        mse_errs.append(mse[1])

    fig, ax1 = plt.subplots()
    ax1.bar([i for i in range(len(rgt_bars))], rgt_bars, width=bar_width, yerr=rgt_errs)
    ax1.set_xticks([r+0.5*bar_width for r in range(len(rgt_bars))], x_labels, fontsize=fontsize)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda yv, _: '{:.0f}%'.format(yv*100)))
    ax1.set_ylabel('Empirical regret', fontsize=fontsize)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(x_title, fontsize=fontsize)

    ax2 = ax1.twinx()
    ax2.bar([i + bar_width for i in range(len(mse_bars))], mse_bars, width=bar_width, yerr=mse_errs, color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax2.set_ylabel('MSE', color="red", fontsize=fontsize)
    ax2.set_ylim(bottom=0)

    for tick in ax1.get_xticklabels():
        tick.set_fontsize(fontsize)
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(fontsize)
    for tick in ax2.get_yticklabels():
        tick.set_fontsize(fontsize)

    fig.tight_layout()

    if fig_file is not None:
        plt.savefig(fig_file, format="pdf")
    plt.show()


# experiment_vary_mse(10, generate_jobs_gamma, "graphs/svf_mse_regret_gamma.pdf")
# experiment_vary_mse(10, generate_jobs_normal, "graphs/svf_mse_regret_normal.pdf")
# experiment_vary_mse(10, generate_jobs_lognormal, "graphs/svf_mse_regret_lognormal.pdf")
# experiment_vary_mse(10, generate_jobs_exponential, "graphs/svf_mse_regret_exponential.pdf")


#####################################################
# # What is the effect of 'larger number of jobs'?
#####################################################
# exp_results = [
#     single_experiment(generate_jobs_gamma, 5, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1),
#     single_experiment(generate_jobs_gamma, 10, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1),
#     single_experiment(generate_jobs_gamma, 15, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1),
#     single_experiment(generate_jobs_gamma, 20, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1),
#     single_experiment(generate_jobs_gamma, 25, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1),
#     single_experiment(generate_jobs_gamma, 30, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1)
# ]
# bar_experiments(exp_results, "nr. of tasks", ["5", "10", "15", "20", "25", "30"], "graphs/svf_nr_jobs.pdf")
#####################################################



#####################################################
# # What is the effect of a low learning sample?
#####################################################
exp_results = [
    single_experiment(generate_jobs_gamma, 10, 10000, lambda x1, x2: 5*x1 + 5*x2, 1),
    single_experiment(generate_jobs_gamma, 10, 1000, lambda x1, x2: 5*x1 + 5*x2, 1),
    single_experiment(generate_jobs_gamma, 10, 100, lambda x1, x2: 5*x1 + 5*x2, 1),
    single_experiment(generate_jobs_gamma, 10, 10, lambda x1, x2: 5*x1 + 5*x2, 1),
]
bar_experiments(exp_results, "nr. of training samples", ["10000", "1000", "100", "10"], "graphs/svf_samples.pdf")
#####################################################
