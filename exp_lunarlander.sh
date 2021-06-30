ENV="LunarLander-v2"
NUM_MODEL="10"
N_EPISODES="1000"
LR="1e-3"

python AveragedDQN_lunarLander.py --gpu_no=0 --seed=0 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN_lunarLander.py --gpu_no=0 --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN_lunarLander.py --gpu_no=0 --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN_lunarLander.py --gpu_no=0 --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN_lunarLander.py --gpu_no=0 --seed=4 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN_lunarLander.py --gpu_no=1 --seed=5 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN_lunarLander.py --gpu_no=1 --seed=6 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR

