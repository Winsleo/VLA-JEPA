#!/bin/bash
# Record the I3 probe dataset: RGB + simulator metric depth clips from on-policy LIBERO rollouts.
#
# Same two-process layout as eval_libero.sh (policy server in the pinned training env, simulator in
# envs/libero_py310), one process pair per suite so each suite writes its own manifest file.
#
# Overridable: SUITES, NUM_TRIALS, SEED, EVAL_GPUS, GEO_CLIPS_OUT, CLIPS_PER_EPISODE, CLIP_STRIDES,
# SEQUENTIAL.
# Smoke run before the full pass:
#   SUITES=libero_goal NUM_TRIALS=1 EVAL_GPUS=0 GEO_CLIPS_OUT=/tmp/geo_clips_smoke bash examples/LIBERO/record_geo_clips.sh
# All four stride variants of the delta-interval sweep, from a single pass through the simulator:
#   CLIP_STRIDES="1 2 4 8" GEO_CLIPS_OUT=/vepfs/wangshilong/data/dynaweave/i3_geo_clips_sweep \
#   bash examples/LIBERO/record_geo_clips.sh

export PYTHONDONTWRITEBYTECODE=1
export LIBERO_HOME=/vepfs/wangshilong/benchmarks/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=/vepfs/wangshilong/envs/libero_py310/bin/python

your_ckpt=/vepfs/wangshilong/models/dynaweave/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt

IFS=' ' read -r -a items <<< "${SUITES:-libero_10 libero_goal libero_object libero_spatial}"
host="127.0.0.1"
# Offset from eval_libero.sh's 15083 so a recording can run next to an evaluation.
base_port=15183
num_trials_per_task=${NUM_TRIALS:-5}
clips_per_episode=${CLIPS_PER_EPISODE:-8}
# Frame strides to cut clips at; each becomes its own dataset root under ${out_root}/s<stride>. One
# rollout serves all of them, so widening this list costs disk and not simulator time.
clip_strides=${CLIP_STRIDES:-1}
with_state="true"
seed=${SEED:-7}
out_root=${GEO_CLIPS_OUT:-/vepfs/wangshilong/data/dynaweave/i3_geo_clips}
IFS=',' read -r -a eval_gpus <<< "${EVAL_GPUS:-4,5,6,7}"

# Suites otherwise run concurrently, one per GPU. On a single-GPU machine that would co-locate every
# policy server and every simulator on one device, so there the default is one suite at a time, with
# each suite's server released before the next starts.
sequential=${SEQUENTIAL:-$([ ${#eval_gpus[@]} -eq 1 ] && echo 1 || echo 0)}

mkdir -p "${out_root}"
index=0

for task_suite_name in "${items[@]}"
do
    port=$((base_port+index))
    cuda=${eval_gpus[$((index % ${#eval_gpus[@]}))]}
    index=$((index+1))

    # Unlike eval_libero.sh the server log goes to a file, so the launching shell is not held open
    # by an inherited stdout that never closes.
    python ./deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16 \
        --cuda ${cuda} > "${out_root}/server_${task_suite_name}.log" 2>&1 &
    server_pid=$!

    recorder=(${sim_python} ./examples/LIBERO/record_geo_clips.py
        --args.pretrained-path ${your_ckpt}
        --args.host "$host"
        --args.port ${port}
        --args.task-suite-name "$task_suite_name"
        --args.num-trials-per-task "$num_trials_per_task"
        --args.clips-per-episode "$clips_per_episode"
        --args.clip-strides ${clip_strides}
        --args.out-path "$out_root"
        --args.with_state "$with_state"
        --args.seed "$seed")

    if [ "${sequential}" = "1" ]; then
        "${recorder[@]}" > "${out_root}/record_${task_suite_name}.log" 2>&1
        kill "${server_pid}" 2>/dev/null || true
    else
        "${recorder[@]}" > "${out_root}/record_${task_suite_name}.log" 2>&1 &
    fi
done

# As in eval_libero.sh, both processes stay in the background: follow ${out_root}/record_*.log for
# progress, and collect the policy servers by PID once every recorder has exited.
jobs -l
