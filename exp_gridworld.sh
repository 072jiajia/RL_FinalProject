ENV='gridworld'
NUM_MODEL='1'
GPU_NO='7'
TOTAL_STEPS='2000000'

python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=0 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=4 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=5 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=6 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=7 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS --lr=$LR