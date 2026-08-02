#!/bin/bash
# Record the I3 probe dataset: RGB + simulator metric depth clips from on-policy LIBERO rollouts.
#
# Same two-process layout as eval_libero.sh (policy server in the pinned training env, simulator in
# envs/libero_py310), one process pair per suite so each suite writes its own manifest file.
#
# Overridable: SUITES, NUM_TRIALS, SEED, EVAL_GPUS, GEO_CLIPS_OUT, CLIPS_PER_EPISODE.
# Smoke run before the full pass:
#   SUITES=libero_goal NUM_TRIALS=1 EVAL_GPUS=0 GEO_CLIPS_OUT=/tmp/geo_clips_smoke bash examples/LIBERO/record_geo_clips.sh

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
with_state="true"
seed=${SEED:-7}
out_root=${GEO_CLIPS_OUT:-/vepfs/wangshilong/data/dynaweave/i3_geo_clips}
IFS=',' read -r -a eval_gpus <<< "${EVAL_GPUS:-4,5,6,7}"

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

    ${sim_python} ./examples/LIBERO/record_geo_clips.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port ${port} \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.clips-per-episode "$clips_per_episode" \
        --args.out-path "$out_root" \
        --args.with_state "$with_state" \
        --args.seed "$seed" > "${out_root}/record_${task_suite_name}.log" 2>&1 &
done

# As in eval_libero.sh, both processes stay in the background: follow ${out_root}/record_*.log for
# progress, and collect the policy servers by PID once every recorder has exited.
jobs -l
