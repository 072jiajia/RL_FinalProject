ENV='Breakout-v0'
NUM_MODEL='5'
TOTAL_FRAMES='120e6'
LR='1e-3'
python AveragedDQN.py --gpu_no=5 --seed=0 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python AveragedDQN.py --gpu_no=5 --seed=1 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python AveragedDQN.py --gpu_no=5 --seed=2 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python AveragedDQN.py --gpu_no=5 --seed=3 --num_model=$NUM_MODEL --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR
