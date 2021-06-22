ENV='Breakout-v0'
NUM_MODEL='5'
TOTAL_FRAMES='120e6'
LR='1e-3'
GPU_NO='4'
python AveragedDQN_atari_v2.py --gpu_no=$GPU_NO --seed=0 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python AveragedDQN_atari_v2.py --gpu_no=$GPU_NO --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR