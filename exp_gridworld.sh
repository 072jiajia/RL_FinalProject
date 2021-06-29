ENV='gridworld'
NUM_MODEL='15'
GPU_NO='9'
TOTAL_STEPS='5000000'

python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=0 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=4 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS &
python AveragedDQN_gridworld.py --gpu_no=$GPU_NO --seed=4 --num_model=$NUM_MODEL --env_name=$ENV --total_steps=$TOTAL_STEPS