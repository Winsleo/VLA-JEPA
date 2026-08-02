#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1
export LIBERO_HOME=/vepfs/wangshilong/benchmarks/LIBERO # your LIBERO code path
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export sim_python=/vepfs/wangshilong/envs/libero_py310/bin/python # your LIBERO conda path

your_ckpt=/vepfs/wangshilong/models/dynaweave/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

items=("libero_10" "libero_goal" "libero_object" "libero_spatial")
host="127.0.0.1"
base_port=15083
unnorm_key="franka"
index=3 # GPUs 0-3 are occupied by another job; README allows adapting the parallelization
# I1-S4 / I2-V6: trial count, seed and GPU list are overridable so the same checkpoint can be
# re-measured without editing the script again. The defaults reproduce the layout above, i.e.
# 50 trials per task on GPUs 4-7, one suite per GPU.
num_trials_per_task=${NUM_TRIALS:-50}
with_state="true"
seed=${SEED:-7}
IFS=',' read -r -a eval_gpus <<< "${EVAL_GPUS:-4,5,6,7}"

# start each task suite on specific GPU.
for task_suite_name in "${items[@]}"
do
    index=$((index+1))
    port=$((base_port+index))
    # Wrap around when fewer GPUs than suites are given, so EVAL_GPUS=0 co-locates all four.
    cuda=${eval_gpus[$(((index - 4) % ${#eval_gpus[@]}))]}

    python ./deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16 \
        --cuda ${cuda} &

    video_out_path="results/${task_suite_name}/${folder_name}_seed${seed}"

    LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
    mkdir -p ${LOG_DIR}
    mkdir -p ${video_out_path}


    # export DEBUG=true

    ${sim_python} ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port ${port}\
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" > "${video_out_path}/eval.log" \
        --args.with_state "$with_state" \
        --args.seed "$seed" &
done