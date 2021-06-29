from matplotlib import pyplot as plt
import json, glob
import numpy as np

# GLOBAL VAR
AXIS_FONT_SIZE = 18 #gridsearch uses 18
LEGEND_FONT_SIZE = 11 #gridsearch uses 11
OPTIMAL = 0

# put all (typically 7) logfiles for each "k", for all seed numbers, in separated folder
LUNARLANDER_OLD = ["log/lunarlander_k1",
               "log/lunarlander_k5",
               "log/lunarlander_k10",
               "log/lunarlander_k15"]
LUNARLANDER_NEW = ["log/new_lunarlander_k1",
               "log/new_lunarlander_k5",
               "log/new_lunarlander_k10",
               "log/new_lunarlander_k15"]
LUNARLANDER = LUNARLANDER_OLD
GRIDWORLD = ['log/gridworld5M_k1',
             'log/gridworld5M_k5',
             'log/gridworld5M_k10',
             'log/gridworld5M_k15']

# Ks = list(range(1, 11)) + [15, 20, 25, 50, 100]
Ks = [1,2,3,4,5]

ROULETTE = ["log/roulette_k" + str(k) for k in Ks] + ["log/roulette_ddqn"]
# plot the avg and std
# label = ['Averaged DQN, K=' + str(K) for K in Ks]
label = ['K=' + str(K) for K in Ks] + ["Double QDN"]

FOLDER_PATH = LUNARLANDER_OLD
if FOLDER_PATH == GRIDWORLD:
    sample_every_n_element = 10
    Ks = [1, 5, 10, 15]
    label = ['K=' + str(K) for K in Ks]
    OPTIMAL = 0.258
elif FOLDER_PATH == LUNARLANDER:
    sample_every_n_element = 10
    Ks = [1, 5, 10, 15]
    label = ['K=' + str(K) for K in Ks]
else:
    sample_every_n_element = None

def main():
    avg_sc_list = []; std_sc_list = []; avg_val_list = []; std_val_list = []; idx_list = []
    for path in FOLDER_PATH:
        avg_score, std_score, avg_val_est, std_val_est, idx = read_log(path, sample_every_n_element=sample_every_n_element)
        avg_sc_list.append(avg_score); std_sc_list.append(std_score)
        avg_val_list.append(avg_val_est); std_val_list.append(std_val_est); idx_list.append(idx)

    plt.figure()
    for i in range(len(avg_sc_list)):
        if i < 10:
            plt.plot(idx_list[i], avg_sc_list[i], label=label[i], linewidth=1)
        else:
            plt.plot(idx_list[i], avg_sc_list[i], ':', label=label[i], linewidth=1)
        plt.fill_between(idx_list[i], avg_sc_list[i] - std_sc_list[i], avg_sc_list[i] + std_sc_list[i], alpha=0.3)
    if FOLDER_PATH == GRIDWORLD:
        plt.plot(idx_list[0], [1]*len(idx_list[0]), '--',label="Optimal", linewidth=2)
        plt.xlabel("Steps (per 10k)", fontsize=AXIS_FONT_SIZE)
    else:
        plt.xlabel("Episode", fontsize=AXIS_FONT_SIZE)
    plt.ylabel("Averaged score", fontsize=AXIS_FONT_SIZE)
    plt.legend(prop={'size': LEGEND_FONT_SIZE}, ncol=2)
    plt.savefig('Roulette1.png')

    plt.figure()
    for i in range(len(avg_val_list)):
        if i < 10:
            plt.plot(idx_list[i], avg_val_list[i], label=label[i], linewidth=1)
        else:
            plt.plot(idx_list[i], avg_val_list[i], ':', label=label[i], linewidth=1)
        plt.fill_between(idx_list[i], avg_val_list[i] - std_val_list[i], avg_val_list[i] + std_val_list[i], alpha=0.3)
    if FOLDER_PATH == GRIDWORLD:
        plt.plot(idx_list[0], [OPTIMAL]*len(idx_list[0]), '--', label="Optimal", linewidth=2)
        plt.xlabel("Steps (per 10k)", fontsize=AXIS_FONT_SIZE)
    else:
        plt.xlabel("Episode", fontsize=AXIS_FONT_SIZE)
    plt.ylabel("Value estimate", fontsize=AXIS_FONT_SIZE)
    plt.legend(prop={'size': LEGEND_FONT_SIZE}, ncol=2)
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    plt.savefig('Roulette2.png')
    plt.show()



def read_log(path, sample_every_n_element=None):
    logfiles = glob.glob(path+"/*.json")
    scores = []; val_est = []; min_len = 9999999999999999
    for logfile in logfiles: # load all logfiles
        with open(logfile) as json_file:
            dict = json.load(json_file)
            temp1 = dict["scores"]
            temp2 = dict["value_est"]
            if (min_len>len(temp1)):
                min_len = len(temp1)
            if (min_len>len(temp2)):
                min_len = len(temp2)
            scores.append(temp1)
            val_est.append(temp2)
    # min_len = min(min_len, 201)
    result_scores = []
    for item in scores: # truncate the data to min_len
        temp1 = np.asarray(item)
        result_scores.append(temp1[:min_len])
    result_scores = np.asarray(result_scores)
    avg_score = np.mean(result_scores, axis=0)
    std_score = np.std(result_scores, axis=0)
    if sample_every_n_element != None:
        avg_score = avg_score[0:avg_score.shape[0]:sample_every_n_element]
        std_score = std_score[0:std_score.shape[0]:sample_every_n_element]

    result_val_est = []
    for item in val_est: # truncate the data to min_len
        temp2 = np.asarray(item)
        result_val_est.append(temp2[:min_len])
    result_val_est = np.asarray(result_val_est)
    avg_val_est = np.mean(result_val_est, axis=0)
    std_val_est = np.std(result_val_est, axis=0)
    idx = np.linspace(0, min_len, min_len)
    if sample_every_n_element != None:
        avg_val_est = avg_val_est[0:avg_val_est.shape[0]:sample_every_n_element]
        std_val_est = std_val_est[0:std_val_est.shape[0]:sample_every_n_element]
        idx = idx[0:idx.shape[0]:sample_every_n_element]
        # idx = idx

    # calculate avg of bias and standard dev from convergence point
    if FOLDER_PATH == GRIDWORLD:
        mask = idx > 300 # evaluate only for those more 3jt training steps
        std_val_converged = std_val_est[mask]
        value_std_converged_mean = np.round(np.mean(std_val_converged),
                                            decimals=3)
        value_est_bias_converged = OPTIMAL - avg_val_est[mask]
        value_est_bias_converged_mean = np.round(np.mean(value_est_bias_converged),
                                                 decimals=3)
        score_bias_converged = 1 - avg_score[mask]
        score_bias_converged_mean = np.round(np.mean(score_bias_converged),
                                             decimals=3)
        print("Path:", path)
        print("value_std_mean: ", value_std_converged_mean, ", val_bias_mean:", value_est_bias_converged_mean,
              ", score_bias mean:", score_bias_converged_mean)
    if FOLDER_PATH == LUNARLANDER:
        mask = idx > 600 # evaluate only for those more 3jt training steps
        std_val_converged = std_val_est[mask]
        value_std_converged_mean = np.round(np.mean(std_val_converged),
                                            decimals=3)
        score_converged = avg_score[mask]
        score_converged_mean = np.round(np.mean(score_converged),
                                             decimals=3)
        print("Path:", path)
        print("value_std_mean: ", value_std_converged_mean,
              ", score mean:", score_converged_mean)
    return avg_score, std_score, avg_val_est, std_val_est, idx

main()