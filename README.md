# RL Group Project: Team 12
Code for ablation project: Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning
All the code files are aready tested, thus, of any error is encountered, please kindly inform to us: `ardianumam.05g@g2.nctu.edu.tw`, or `jia.cs07@nycu.edu.tw`, or `huangsinfu91212@nctu.edu.tw`
## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
    (optional) virtualenv .
    (optional) source bin/activate
    pip install -r requirements.txt
```
To install the atari games, you may follow the steps [here](https://github.com/openai/atari-py)

## Reproducing our results
```
    sh exp_{gridworld, lunarlander, roulette}.sh
```

**Note**: 
* run `sed -i 's/\r$//' exp.sh` before, to remove the '\r' generated in Windows (this causes error).
* one of succesful ways to install gym-atari: [here](https://github.com/openai/gym/issues/1218)


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
