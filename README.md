

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
    --> Better to run ".sh" file (see exp_breakout.sh for example) 
```
## TODO:
1. Perform experiment in some envs (WIP)
* Seaquest-v0 (state = [210, 160, 3], action = [18]) --> not yet  
* LunarLander-v2 (state = [8], action = [4]) --> WIP
* Asterix-v0 (state = [210, 160, 3], action = [9]) --> not yet
* Breakout-v0 (state = [210, 160, 3], action = [4]) --> WIP

2. Log the average value (DONE)
3. Recency weighting version

**Note**: 
* run `sed -i 's/\r$//' exp.sh` before, to remove the '\r' generated in Windows (this causes error).
* one of succesful ways to install gym-atari: [here](https://github.com/openai/gym/issues/1218)


