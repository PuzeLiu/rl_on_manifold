import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def get_mean_and_confidence(data):
    """
    Compute the mean and 95% confidence interval
    Args:
        data (np.ndarray): Array of experiment data of shape (n_runs, n_epochs).
    Returns:
        The mean of the dataset at each epoch along with the confidence interval.
    """
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval, _ = st.t.interval(0.95, n - 1, scale=se)
    return mean, interval


def get_data_list(file_dir, prefix):
    data = list()
    for filename in os.listdir(file_dir):
        if filename.startswith(prefix+'-'):
            data.append(np.load(os.path.join(file_dir, filename), allow_pickle=True))
    return np.array(data)


def plot_learning_curve(log_dir, exp_list):
    cm = plt.get_cmap('tab10')

    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 10))
    fig1.subplots_adjust(hspace=.5)
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 10))
    fig2.subplots_adjust(hspace=.5)

    for i, exp_suffix in enumerate(exp_list):
        color_i = cm(i)
        exp_dir = os.path.join(log_dir, exp_suffix)
        _, seeds, _ = next(os.walk(exp_dir))

        exp_J_list = get_data_list(exp_dir, "J")
        exp_R_list = get_data_list(exp_dir, "R")
        exp_E_list = get_data_list(exp_dir, "E")
        c_max_list = get_data_list(exp_dir, "c_max")
        c_avg_list = get_data_list(exp_dir, "c_avg")
        c_dq_max_list = get_data_list(exp_dir, "c_dq_max")

        mean_J, conf_J = get_mean_and_confidence(exp_J_list)
        mean_R, conf_R = get_mean_and_confidence(exp_R_list)

        ax1[0].plot(mean_J, label=exp_suffix, color=color_i)
        ax1[0].fill_between(np.arange(np.size(mean_J)), mean_J + conf_J, mean_J - conf_J, alpha=0.2, color=color_i)
        ax1[0].set_title("J")
        ax1[1].plot(mean_R, label=exp_suffix, color=color_i)
        ax1[1].fill_between(np.arange(np.size(mean_R)), mean_R + conf_R, mean_R - conf_R, alpha=0.2, color=color_i)
        ax1[1].set_title("R")
        ax1[1].legend()

        if np.all(exp_E_list!=None):
            mean_E, conf_E = get_mean_and_confidence(exp_E_list[:, 1:])
            ax1[2].plot(mean_E, label=exp_suffix, color=color_i)
            ax1[2].fill_between(np.arange(np.size(mean_E)), mean_E + conf_E, mean_E - conf_E, alpha=0.2, color=color_i)
            ax1[2].set_title("E")

        mean_c_max, conf_c_max = get_mean_and_confidence(c_max_list)
        mean_c_avg, conf_c_avg = get_mean_and_confidence(c_avg_list)
        mean_c_dq_max, conf_c_dq_max = get_mean_and_confidence(c_dq_max_list)
        ax2[0].plot(mean_c_max, label=exp_suffix, color=color_i)
        ax2[0].fill_between(np.arange(np.size(mean_c_max)), mean_c_max + conf_c_max,
                            mean_c_max - conf_c_max, alpha=0.2, color=color_i)
        ax2[0].set_title("c_max")
        ax2[1].plot(mean_c_avg, label=exp_suffix, color=color_i)
        ax2[1].fill_between(np.arange(np.size(mean_c_avg)), mean_c_avg + conf_c_avg,
                            mean_c_avg - conf_c_avg, alpha=0.2, color=color_i)
        ax2[1].set_title("c_avg")
        ax2[2].plot(mean_c_dq_max, label=exp_suffix, color=color_i)
        ax2[2].fill_between(np.arange(np.size(mean_c_dq_max)), mean_c_dq_max + conf_c_dq_max,
                            mean_c_dq_max - conf_c_dq_max, alpha=0.2, color=color_i)
        ax2[2].set_title("c_dq_max")
        ax2[2].legend()

    fig1.savefig(os.path.join(log_dir, "Reward.pdf"))
    fig2.savefig(os.path.join(log_dir, "Constraints.pdf"))
    plt.show()


def plot_learning_metric(log_dir, exp_list, metric, label_list, title, y_scale='linear', file_name=None):
    cm = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(12, 9))
    ax = plt.gca()

    for i, exp_suffix in enumerate(exp_list):
        color_i = cm(i)
        exp_dir = os.path.join(log_dir, exp_suffix)
        _, seeds, _ = next(os.walk(exp_dir))

        metric_list = get_data_list(exp_dir, str(metric))

        mean_metric, conf_metric = get_mean_and_confidence(metric_list)

        ax.plot(mean_metric, label=label_list[i], color=color_i)
        ax.fill_between(np.arange(np.size(mean_metric)), mean_metric + conf_metric, mean_metric - conf_metric,
                        alpha=0.1, color=color_i)
        ax.legend(fontsize=30)
        ax.set_yscale(y_scale)

    ax.set_title(title, fontsize=40)
    ax.tick_params('both', labelsize=30)
    if file_name is None:
        file_name = title + ".pdf"
    else:
        file_name += ".pdf"
    fig.savefig(os.path.join(log_dir, file_name))
    plt.show()

def plot_learning_curve_single(log_dir, exp_name, seeds):
    fig1, ax1 = plt.subplots(3, 1)
    fig1.subplots_adjust(hspace=.5)
    fig2, ax2 = plt.subplots(3, 1)
    fig2.subplots_adjust(hspace=.5)

    exp_dir = os.path.join(log_dir, exp_name)

    for seed in seeds:
        postfix = "-" + str(seed) + ".npy"
        J = np.load(os.path.join(exp_dir, "J" + postfix))
        R = np.load(os.path.join(exp_dir, "R" + postfix))
        E = np.load(os.path.join(exp_dir, "E" + postfix), allow_pickle=True)
        c_max = np.load(os.path.join(exp_dir, "c_max" + postfix))
        c_avg = np.load(os.path.join(exp_dir, "c_avg" + postfix))
        c_dq_max = np.load(os.path.join(exp_dir, "c_dq_max" + postfix))

        ax1[0].plot(J)
        ax1[0].set_title("J")
        ax1[1].plot(R)
        ax1[1].set_title("R")

        if np.all(E!=None):
            ax1[2].plot(E)
            ax1[2].set_title("E")

        ax2[0].plot(c_max)
        ax2[0].plot(np.zeros_like(c_max), c='tab:red', lw=2)
        ax2[0].set_title("c_max")
        ax2[1].plot(c_avg)
        ax2[1].plot(np.zeros_like(c_avg), c='tab:red', lw=2)
        ax2[1].set_title("c_avg")
        ax2[2].plot(c_dq_max)
        ax2[2].plot(np.zeros_like(c_dq_max), c='tab:red', lw=2)
        ax2[2].set_title("c_dq_max")

    plt.show()