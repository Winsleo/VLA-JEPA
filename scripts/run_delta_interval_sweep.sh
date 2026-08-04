#!/bin/bash
# Drive the I3 delta-interval sweep: one (recording stride, delta lag) cell at a time, resumable.
#
# One delta target spans `TUBELET_SIZE * lag * stride` frames, i.e. `0.1 * lag * stride` seconds at
# LIBERO's 20 Hz, so the two axes reach the same duration by different routes: with the teacher seeing
# the frames in between (small stride, large lag) and without (large stride, small lag). Those pairs
# are the point of the sweep rather than duplicates -- they separate "the target got easier" from
# "the teacher's input changed" (`docs/experiments/i3-geo-probes.md` section 9).
#
# Each stride gets its own probe cache root: `probe_cache.features_dir` puts only the arm in the path,
# and features extracted from stride-2 clips are different features, not a variant of the same ones.
# Lags share a stride's root, because `targets_dir` does encode the lag.
#
# Overridable: STRIDES, LAGS, ARMS, CLIPS_ROOT, CACHE_ROOT, REPORTS, LOGS, PY, DEVICE, FIGURE.
# A cell whose report already exists is skipped, so an interrupted sweep resumes where it stopped.
#
# Lag-only sweep on an existing clip cache (no re-recording, no new features):
#   STRIDES=1 CLIPS_ROOT=/vepfs/wangshilong/data/dynaweave \
#   CACHE_ROOT=/vepfs/wangshilong/data/dynaweave bash scripts/run_delta_interval_sweep.sh

set -euo pipefail
# The pinned environment is selected by absolute path, so a stray project env must not shadow it.
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV

PY=${PY:-/vepfs/wangshilong/envs/dynaweave/bin/python}
STRIDES=${STRIDES:-"1 2 4 8"}
LAGS=${LAGS:-"1 2 3"}
# The pre-registered primary pair; both are @256, so a cell varies the teacher and nothing else.
ARMS=${ARMS:-"A D"}
CLIPS_ROOT=${CLIPS_ROOT:-/vepfs/wangshilong/data/dynaweave/i3_geo_clips_sweep}
CACHE_ROOT=${CACHE_ROOT:-/vepfs/wangshilong/data/dynaweave/i3_probe_cache_sweep}
REPORTS=${REPORTS:-${CACHE_ROOT}/reports}
LOGS=${LOGS:-${CACHE_ROOT}/logs}
DEVICE=${DEVICE:-cuda}
# Written next to the table, in the data tree: the results document lives in the superproject, so
# copying the chosen figure into it is a documentation step rather than something this script reaches
# across the submodule boundary to do.
FIGURE=${FIGURE:-${REPORTS}/i3-delta-interval}

cd "$(dirname "$0")/.."
mkdir -p "${REPORTS}" "${LOGS}"

for stride in ${STRIDES}; do
    clips=${CLIPS_ROOT}/s${stride}
    out=${CACHE_ROOT}/s${stride}
    if [ ! -d "${clips}" ]; then
        echo "missing clip cache ${clips}: record it with record_geo_clips.sh CLIP_STRIDES first" >&2
        exit 1
    fi

    # Skips itself when the cache is already complete, so a resumed sweep pays no forward passes.
    echo "=== s${stride}: features for arms ${ARMS} ==="
    ${PY} scripts/run_geo_probes.py cache --clips "${clips}" --out "${out}" --arms ${ARMS} \
        --device "${DEVICE}" >"${LOGS}/cache_s${stride}.log" 2>&1

    for lag in ${LAGS}; do
        report=${REPORTS}/geo_probes_s${stride}_lag${lag}.json
        if [ -f "${report}" ]; then
            echo "=== s${stride} lag${lag}: ${report} exists, skipping ==="
            continue
        fi

        echo "=== s${stride} lag${lag}: targets ==="
        ${PY} scripts/run_geo_probes.py targets --clips "${clips}" --out "${out}" \
            --delta-lag "${lag}" >"${LOGS}/targets_s${stride}_lag${lag}.log" 2>&1

        # The state probe does not depend on the lag, so it is fitted once per stride, at the first
        # lag: it is the check that a wider recording stride did not damage the state result itself.
        kinds="delta"
        [ "${lag}" = "${LAGS%% *}" ] && kinds="state delta"
        echo "=== s${stride} lag${lag}: fit ${kinds} ==="
        ${PY} scripts/run_geo_probes.py fit --clips "${clips}" --out "${out}" --arms ${ARMS} \
            --kinds ${kinds} --delta-lag "${lag}" --device "${DEVICE}" --report "${report}" \
            >"${LOGS}/fit_s${stride}_lag${lag}.log" 2>&1
    done
done

echo "=== curve ==="
${PY} scripts/run_geo_probes.py sweep --reports "${REPORTS}" --arms ${ARMS} \
    --markdown "${REPORTS}/i3-delta-interval.md" --figure "${FIGURE}"
