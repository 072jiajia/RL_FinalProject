# RL Group Project: Team 12
Code for ablation project: Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning<br>
All the code files are aready tested, thus, if any error is encountered, please kindly confirm to us: `ardianumam.05g@g2.nctu.edu.tw`, `jia.cs07@nycu.edu.tw`, `huangsinfu91212@nctu.edu.tw`. This code is mantained in this [Github repo](https://github.com/072jiajia/RL_FinalProject/tree/cleaned).
## Reports
A detailed report of this repo can be read [here](https://drive.google.com/file/d/1sDnAcZXfYWPPwjxvOyLDQd3KW5r0trEG/view?usp=sharing).
## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
    (optional) virtualenv .
    (optional) source bin/activate
    pip install -r requirements.txt
```
To install the atari games, you may follow the steps [here](https://github.com/openai/atari-py).
However, we don't provide the experiment for Atari due to resource limitation. But, an experiment can still be conducted, if needed, using
`AveragedDQN_atari.py` (more detailed explanation is in our report).

## Reproducing our results
```
    sh exp_{gridworld, lunarlander, roulette}.sh
```

**Note**: 
* It may need to run `sed -i 's/\r$//' exp*.sh` before, to remove the '\r' due to different encoding in different OS (this causes error).
* If error is encountered during Atari installation, you may refer [this](https://github.com/openai/gym/issues/1218).
* If running bash script above encounters an issue, you may run directly the script by: `python AveragedDQN_*.py`

## Plotting The Results
Our results are already dumped in to json files in `log` directory.
To draw the plots, please move the json files to the folders named log/{env's name}\_k{number of K}
And plot our result by:<br>
```
python plot_trainlog.py --env=<env_name>
```
`env_name` options are: gridwolrd, roulette or lunarlander


## Citation
```
@inproceedings{anschel2017averaged,
  title={Averaged-dqn: Variance reduction and stabilization for deep reinforcement learning},
  author={Anschel, Oron and Baram, Nir and Shimkin, Nahum},
  booktitle={International Conference on Machine Learning},
  pages={176--185},
  year={2017},
  organization={PMLR}
}
```
