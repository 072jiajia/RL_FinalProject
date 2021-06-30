ENV='Roulette-v0'
TOTAL_FRAMES='1e6'
LR='1e-3'
GPU_NO='4'
AVERAGEDDQN_FILE='AveragedDQN_roulette.py'
DDQN_FILE='DDQN_roulette.py'

call_function(){
    python $1 --gpu_no=$2 --seed=1 --num_model=$3 --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
    python $1 --gpu_no=$2 --seed=2 --num_model=$3 --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
    python $1 --gpu_no=$2 --seed=3 --num_model=$3 --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR
}

call_function $AVERAGEDDQN_FILE 1 1 &
call_function $AVERAGEDDQN_FILE 2 2 &
call_function $AVERAGEDDQN_FILE 3 3 &
call_function $AVERAGEDDQN_FILE 4 4 &
call_function $AVERAGEDDQN_FILE 5 5
call_function $AVERAGEDDQN_FILE 6 6 &
call_function $AVERAGEDDQN_FILE 7 7 &
call_function $AVERAGEDDQN_FILE 8 8 &
call_function $AVERAGEDDQN_FILE 9 9 &
call_function $AVERAGEDDQN_FILE 1 10
call_function $AVERAGEDDQN_FILE 2 15 &
call_function $AVERAGEDDQN_FILE 3 20 &
call_function $AVERAGEDDQN_FILE 4 25 &
call_function $AVERAGEDDQN_FILE 5 50
call_function $AVERAGEDDQN_FILE 6 100 &
python $DDQN_FILE --gpu_no=7 --seed=1 --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python $DDQN_FILE --gpu_no=7 --seed=2 --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR &
python $DDQN_FILE --gpu_no=7 --seed=3 --env_name=$ENV --total_frames=$TOTAL_FRAMES --lr=$LR