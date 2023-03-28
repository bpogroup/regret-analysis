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


# y_i = f(x_i)
# \hat{y_i} = y_i + e_i, where e_i has bais \mu and variance \sigma^2_i + \nu^2_i
# \sigma^2_i is the irreducible error
# \nu^2_i is the prediction error


# Example Arik:
# y_i ~ N(5, 1)
# \hat{y_i} ~ y_i + N(0, \sqrt{\sigma^2_i + \nu^2_1}), where \sigma^2_i = 1 and \nu^2_i in [1, 10]


# Approach:
# 1. Sample some x_i.
# 2. Generate true y_i = f(x_i) + \epsilon for different epsilon, using f(x) = a \cdot x + b
# 3. Learn \hat{f} through x_i, y_i combinations
# 4. Calculate \hat{MSE} on y_i, \hat{f}(x_i). This should equal \sigma^2 of \epsilon.
# 5. Now also calculate the regret.
# NOTE: we are not using a holdout test set here, but that is okay for the experiment.

# Now we do the following:
# 2. use f(x) = a \cdot x^2 + b to generate true y_i
# 3. learn \hat{f} as a linear function
# 4. Now \hat{MSE} > \sigma^2 of epsilon. The difference is \nu^2.

# Repeat for different values of a, b and \sigma^2.

def generate_jobs_normal(n, sigma2, f):
    """
    Generates n jobs.
    Returns the list of jobs, with each job is a tuple (x1, x2, y)
    """
    jobs = []
    for i in range(n):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)

        y = f(x1, x2) + random.normalvariate(5, sqrt(sigma2))

        jobs.append((x1, x2, y))

    return jobs


def generate_jobs_gamma(n, sigma2, f):
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


def generate_jobs_different_errors(n, sigma2, f):
    """
    Generates n jobs.
    Returns the list of jobs, with each job is a tuple (x1, x2, y)
    """
    jobs = []
    for i in range(n):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)

        if x1 < 0.5:
            mod_sigma2 = sigma2 * 2
        else:
            mod_sigma2 = sigma2 / 2

        k = 1 / 4  # keep fixed to keep the skewness fixed
        theta = sqrt(mod_sigma2 / k)  # theta = sqrt(sigma^2/k)
        m = k * theta  # mean = k * theta
        error = scipy.stats.gamma.rvs(k, scale=theta) - m  # deduct the mean to keep the error around 0

        y = f(x1, x2) + error

        jobs.append((x1, x2, y))

    return jobs


def xs(jobs):
    return np.array([[x1, x2] for (x1, x2, y) in jobs])


def ys(jobs):
    return np.array([y for (x1, x2, y) in jobs])


def learn_f_hat(jobs):
    return LinearRegression().fit(xs(jobs), ys(jobs))


def predict_f_hat(model, jobs):
    return model.predict(xs(jobs))


def learn_f_hat_mlp(jobs):
    return MLPRegressor().fit(xs(jobs), ys(jobs))


def learn_f_hat_wo_x2(jobs):
    return LinearRegression().fit(np.array(xs(jobs))[:, 0].reshape(-1, 1), ys(jobs))


def predict_f_hat_wo_x2(model, jobs):
    return model.predict(np.array(xs(jobs))[:, 0].reshape(-1, 1))


def cost_sct(tau, job_pts):
    result = 0
    waiting_time = 0
    for i in range(len(tau)):
        result += waiting_time + job_pts[tau[i]]
        waiting_time += job_pts[tau[i]]
    return result


def schedule_spt(job_pts):
    index_x_job = [(i, job_pts[i]) for i in range(len(job_pts))]
    index_x_job.sort(key=lambda x: x[1])
    return [job[0] for job in index_x_job]


def regret(actual_job_pts, predicted_job_pts, f_scheduling, f_cost):
    # schedule the jobs according to the predicted processing times
    schedule_optimal = f_scheduling(actual_job_pts)
    costs_optimal = f_cost(schedule_optimal, actual_job_pts)

    schedule_predicted = f_scheduling(predicted_job_pts)
    costs_actual = f_cost(schedule_predicted, actual_job_pts)

    return costs_actual - costs_optimal


def line_with_ci(series, series2, x_lab, y_lab, y2_lab, fig_file=None):
    fig, ax1 = plt.subplots()

    x = series.keys()
    y = [m for (m, h) in series.values()]
    ci_bottom = [m-h for (m, h) in series.values()]
    ci_top = [m+h for (m, h) in series.values()]

    ax1.plot(x, y)
    ax1.fill_between(x, ci_bottom, ci_top, color='blue', alpha=0.1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda yv, _: '{:.1f}%'.format(yv*100)))
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


def probability_distribution(jobs):
    sns.kdeplot(ys(jobs), bw_method=0.25)
    plt.show()


def estimate_regret(job_creation_function, model, nr_jobs, sigma2, nr_experiments, f_y, f_hat_predicter):
    regrets = []
    for _ in range(nr_experiments):
        jobs = job_creation_function(nr_jobs, sigma2, f_y)
        y_hats = f_hat_predicter(model, jobs)

        regrets.append(regret(ys(jobs), y_hats, schedule_spt, cost_sct))

    avg = mean(regrets)
    ci = st.t.interval(0.95, len(regrets) - 1, loc=avg, scale=st.sem(regrets))

    return avg, avg - ci[0]


def experiment(job_creation_function, nr_tasks, nr_learning_samples, f_y, f_hat_learner=learn_f_hat, f_hat_predicter=predict_f_hat):
    sigma2_x_regret = dict()
    sigma2_x_mse = dict()
    for sigma2 in [s/10 for s in range(1, 10)] + [s/10 for s in range(10, 50, 5)] + [s/10 for s in range(50, 110, 10)]:
        # 1. Sample some x_i.
        # 2. Generate true y_i = f(x_i) + \epsilon for different epsilon, using f(x) = a \cdot x + b
        jobs = job_creation_function(nr_learning_samples, sigma2, f_y)
        # 3. Learn \hat{f} through x_i, y_i combinations
        model = f_hat_learner(jobs)
        # 4. Calculate \hat{MSE} on y_i, \hat{f}(x_i). This should equal \sigma^2 of \epsilon.
        jobs = job_creation_function(20000, sigma2, f_y)
        mse = mean_squared_error(ys(jobs), f_hat_predicter(model, jobs))
        print(sigma2, mse)
        # 5. Now also calculate the regret.
        rgt = estimate_regret(job_creation_function, model, nr_tasks, sigma2, 10000, f_y, f_hat_predicter)

        sigma2_x_regret[sigma2] = rgt
        sigma2_x_mse[sigma2] = mse

    return sigma2_x_regret, sigma2_x_mse


def single_experiment(job_creation_function, nr_tasks, nr_learning_samples, f_y, sigma2, f_hat_learner=learn_f_hat, f_hat_predicter=predict_f_hat):
    # 1. Sample some x_i.
    # 2. Generate true y_i = f(x_i) + \epsilon for different epsilon, using f(x) = a \cdot x + b
    jobs = job_creation_function(nr_learning_samples, sigma2, f_y)
    # 3. Learn \hat{f} through x_i, y_i combinations
    model = f_hat_learner(jobs)
    # 4. Calculate \hat{MSE} on y_i, \hat{f}(x_i). This should equal \sigma^2 of \epsilon.
    jobs = job_creation_function(25000, sigma2, f_y)
    mse = mean_squared_error(ys(jobs), f_hat_predicter(model, jobs))
    print(sigma2, mse)
    # 5. Now also calculate the regret.
    rgt = estimate_regret(job_creation_function, model, nr_tasks, sigma2, 10000, f_y, f_hat_predicter)

    return (mse, 0), rgt


def bar_experiments(experiment_results, x_labels, fig_file=None):
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
    # ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda yv, _: '{:.1f}%'.format(yv*100)))
    ax1.set_ylabel('Empirical regret', fontsize=fontsize)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.bar([i + bar_width for i in range(len(mse_bars))], mse_bars, width=bar_width, color="red")
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


#####################################################
# # Probability distribution of processing times.
# jobs = generate_jobs_gamma(10000, 5, lambda x1, x2: 5)
# print(mean(ys(jobs)))
# probability_distribution(jobs)

#####################################################
# # How do regret/ MSE develop depending on sigma^2?
#####################################################
# sigma2_x_regret, sigma2_x_mse = experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: 5)
# line_with_ci(sigma2_x_regret, sigma2_x_mse, r'$\sigma^2$', 'regret', 'mse', "graphs/sct_sigma_regret.pdf")
# # This relation is robust for a different number of jobs.
# sigma2_x_regret, sigma2_x_mse = experiment(generate_jobs_gamma, 10, 10000, lambda x1, x2: 5)
# line_with_ci(sigma2_x_regret, sigma2_x_mse, r'$\sigma^2$', 'regret', 'mse')
# # This relation is robust for different error distribution for each 'task'.
# sigma2_x_regret, sigma2_x_mse = experiment(generate_jobs_different_errors, 3, 10000, lambda x1, x2: 5)
# line_with_ci(sigma2_x_regret, sigma2_x_mse, r'$\sigma^2$', 'regret', 'mse')
# # This relation is robust when different tasks have very different processing times.
# sigma2_x_regret, sigma2_x_mse = experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: round(x1)*4 + 4, learn_f_hat_mlp)
# line_with_ci(sigma2_x_regret, sigma2_x_mse, r'$\sigma^2$', 'regret', 'mse')
#####################################################


#####################################################
# # What is the effect of a low learning sample?
#####################################################
exp_results = [
    single_experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: 5*x1 + 5*x2, 1),
    single_experiment(generate_jobs_gamma, 3, 1000, lambda x1, x2: 5*x1 + 5*x2, 1),
    single_experiment(generate_jobs_gamma, 3, 100, lambda x1, x2: 5*x1 + 5*x2, 1),
    single_experiment(generate_jobs_gamma, 3, 10, lambda x1, x2: 5 * x1 + 5 * x2, 1),
]
bar_experiments(exp_results, ["10000", "1000", "100", "10"], "graphs/a_samples.pdf")
#####################################################


#####################################################
# # What is the effect of under-fitting?
#####################################################
# exp_results = [
#     single_experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: 5*x1**2 + 5*x2**2, 1, learn_f_hat_mlp),
#     single_experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: 5*x1**2 + 5*x2**2, 1),
# ]
# bar_experiments(exp_results, ["baseline", "underfitted"], "graphs/b_underfitting.pdf")
#####################################################


#####################################################
# # What is the effect of 'missing features'?
#####################################################
# exp_results = [
#     single_experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: 5*x1 + 5*x2, 1),
#     single_experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: 5*x1 + 5*x2, 1, learn_f_hat_wo_x2, predict_f_hat_wo_x2),
# ]
# bar_experiments(exp_results, ["baseline", "missing features"], "graphs/c_missing_features.pdf")
#####################################################

#####################################################
# # What is the effect of 'larger number of jobs'?
#####################################################
# exp_results = [
#     single_experiment(generate_jobs_gamma, 3, 10000, lambda x1, x2: 5*x1 + 5*x2, 1),
#     single_experiment(generate_jobs_gamma, 6, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1),
#     single_experiment(generate_jobs_gamma, 9, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1),
#     single_experiment(generate_jobs_gamma, 12, 10000, lambda x1, x2: 5 * x1 + 5 * x2, 1)
# ]
# bar_experiments(exp_results, ["3", "6", "9", "12"], "graphs/d_nr_jobs.pdf")
#####################################################
