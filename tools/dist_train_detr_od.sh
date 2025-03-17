#!/usr/bin/env bash
TYPE=$1
FOLD=1
PERCENT=10
GPUS=1


rangeStart=29590
rangeEnd=29690

PORT=0
# 判断当前端口是否被占用，没被占用返回0，反之1
function Listening {
   TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
   UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
   (( Listeningnum = TCPListeningnum + UDPListeningnum ))
   if [ $Listeningnum == 0 ]; then
       echo "0"
   else
       echo "1"
   fi
}

# 指定区间随机数
function random_range {
   shuf -i $1-$2 -n1
}

# 得到随机端口
function get_random_port {
   templ=0
   while [ $PORT == 0 ]; do
       temp1=`random_range $1 $2`
       if [ `Listening $temp1` == 0 ] ; then
              PORT=$temp1
       fi
   done
   echo "Using Port=$PORT"
}
# Function to get GPU memory usage
function get_gpu_memory {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
}

# Start the timer
start_time=$(date +%s)

# Log initial GPU memory usage
initial_gpu_memory=$(get_gpu_memory)
echo "Initial GPU Memory Usage: $initial_gpu_memory MB"


# main
get_random_port ${rangeStart} ${rangeEnd};


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
if [[ ${TYPE} == 'dino_detr' ]]; then
    WORK_DIR='work_dirs_DocLaynet_od/dino_detr_r50_8x2_12e_coco'
    mkdir -p $WORK_DIR
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT \
    $(dirname "$0")/train_detr_od.py configs/dino_detr/dino_detr_r50_8x2_12e_coco.py \
    --launcher pytorch \
    --work-dir $WORK_DIR

fi
# End the timer
end_time=$(date +%s)

# Log final GPU memory usage
final_gpu_memory=$(get_gpu_memory)
echo "Final GPU Memory Usage: $final_gpu_memory MB"

# Calculate and print training time
training_time=$((end_time - start_time))
echo "Training Time: $training_time seconds"


# Save training_time and final_gpu_memory in a result file
echo "Training Time: $training_time seconds" >> training_doclayall.txt
echo "Final GPU Memory Usage: $final_gpu_memory MB" >> training_doclayall.txt

