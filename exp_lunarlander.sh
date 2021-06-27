ENV='LunarLander-v2'
NUM_MODEL='15'
GPU_NO='4'
TOTAL_FRAMES='1000000'

python AveragedDQN_lunarLander.py --gpu_no=$GPU_NO --seed=0 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES &
python AveragedDQN_lunarLander.py --gpu_no=$GPU_NO --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES &
python AveragedDQN_lunarLander.py --gpu_no=$GPU_NO --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES &
python AveragedDQN_lunarLander.py --gpu_no=$GPU_NO --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES &
python AveragedDQN_lunarLander.py --gpu_no=$GPU_NO --seed=4 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES