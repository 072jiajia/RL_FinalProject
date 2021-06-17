from matplotlib import pyplot as plt
import json, glob
import numpy as np

# put all (typically 7) logfiles for each "k", for all seed numbers, in separated folder
FOLDER_PATH = ["log/lunarlender_k5"]

def main():
    avg_list = []; std_list = []; idx_list = []
    for path in FOLDER_PATH:
        avg, std, idx = read_log(path)
        avg_list.append(avg); std_list.append(std); idx_list.append(idx)

    # plot the avg and std
    plt.figure()
    for i in range(len(avg_list)):
        plt.plot(idx_list[i], avg_list[i], label='DQN-'+str(i))
        plt.fill_between(idx_list[i], avg_list[i] - std_list[i], avg_list[i] + std_list[i], alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Averaged score")
    plt.legend()
    plt.show()



def read_log(path):
    logfiles = glob.glob(path+"/*.json")
    data = []; min_len = 9999999999999999
    for logfile in logfiles: # load all logfiles
        with open(logfile) as json_file:
            temp = json.load(json_file)["scores"]
            if (min_len>len(temp)):
                min_len = len(temp)
            data.append(temp)
    result = []
    for item in data: # truncate the data to min_len
        temp = np.asarray(item)
        result.append(temp[:min_len])
    result = np.asarray(result)
    avg = np.mean(result, axis=0)
    std = np.std(result, axis=0)
    idx = np.linspace(0, min_len, min_len)
    return avg, std, idx

main()