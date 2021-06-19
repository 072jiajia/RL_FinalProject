

This code was edited from https://github.com/iankgoode/DQN-for-LunarLander-v2/blob/master/dqn.ipynb


## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
    (optional) virtualenv .
    (optional) source bin/activate
    pip install -r requirements.txt
```

## Run the code
```
    python AveragedDQN.py --gpu_no=5 --num_model=5 --seed=0 --n_episodes=1000 --env_name=LunarLander-v2
    python DQN.py
```
TODO:
1. Experiment using env used in the paper (WIP)
2. Log the average value (DONE)
3. Recency weighting version

.sh


