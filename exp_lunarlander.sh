ENV='LunarLander-v2'
NUM_MODEL='1'
GPU_NO='6'
N_EPISODES='1000'
LR='1e-3'
python AveragedDQN.py --gpu_no=$GPU_NO --seed=0 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=$GPU_NO --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=$GPU_NO --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=$GPU_NO --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=$GPU_NO --seed=4 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=$GPU_NO --seed=5 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=$GPU_NO --seed=6 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR &
python AveragedDQN.py --gpu_no=$GPU_NO --seed=7 --num_model=$NUM_MODEL --env_name=$ENV --n_episodes=$N_EPISODES --lr=$LR