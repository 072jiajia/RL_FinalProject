ENV=LunarLander-v2
NUM_MODEL=5
N_EPISODES=1000
LR=1e-3
python AveragedDQN.py --gpu_no=4 --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=4 --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=4 --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR