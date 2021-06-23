ENV='Asterix-v0'
NUM_MODEL='2'
TOTAL_FRAMES='120e6'
LR='1e-3'
GPU_NO='4'
python AveragedDQN_atari_v3.py --gpu_no=1 --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python AveragedDQN_atari_v3.py --gpu_no=2 --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python AveragedDQN_atari_v3.py --gpu_no=3 --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR
