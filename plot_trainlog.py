from matplotlib import pyplot as plt
import json, glob
import numpy as np

# put all (typically 7) logfiles for each "k", for all seed numbers, in separated folder
FOLDER_PATH = ["log/lunarlander_k1",
               "log/lunarlander_k5",
               "log/lunarlander_k10",
               "log/lunarlander_k15"]

def main():
    avg_sc_list = []; std_sc_list = []; avg_val_list = []; std_val_list = []; idx_list = []
    for path in FOLDER_PATH:
        avg_score, std_score, avg_val_est, std_val_est, idx = read_log(path)
        avg_sc_list.append(avg_score); std_sc_list.append(std_score)
        avg_val_list.append(avg_val_est); std_val_list.append(std_val_est); idx_list.append(idx)

    # plot the avg and std
    label = ['Averaged DQN, K=1',
             'Averaged DQN, K=5',
             'Averaged DQN, K=10',
             'Averaged DQN, K=15',]
    plt.figure()
    plt.title("Average score")
    for i in range(len(avg_sc_list)):
        plt.plot(idx_list[i], avg_sc_list[i], ':', label=label[i], linewidth=1)
        plt.fill_between(idx_list[i], avg_sc_list[i] - std_sc_list[i], avg_sc_list[i] + std_sc_list[i], alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Averaged score")
    plt.legend()

    plt.figure()
    plt.title("Value estimate")
    for i in range(len(avg_val_list)):
        plt.plot(idx_list[i], avg_val_list[i], ':', label=label[i], linewidth=1)
        plt.fill_between(idx_list[i], avg_val_list[i] - std_val_list[i], avg_val_list[i] + std_val_list[i], alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Averaged score")
    plt.legend()

    plt.show()



def read_log(path):
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
    result_scores = []; result_val_est = []
    for item in scores: # truncate the data to min_len
        temp1 = np.asarray(item)
        result_scores.append(temp1[:min_len])
    result_scores = np.asarray(result_scores)
    avg_score = np.mean(result_scores, axis=0)
    std_score = np.std(result_scores, axis=0)

    for item in val_est: # truncate the data to min_len
        temp2 = np.asarray(item)
        result_val_est.append(temp2[:min_len])
    result_val_est = np.asarray(result_val_est)
    avg_val_est = np.mean(result_val_est, axis=0)
    std_val_est = np.std(result_val_est, axis=0)

    idx = np.linspace(0, min_len, min_len)
    return avg_score, std_score, avg_val_est, std_val_est, idx

main()