#!/bin/bash
# Start the Verne V0.5 policy server. eval.sh calls it.
set -euo pipefail

bench_name=$1 task_name=$2 checkpoint=$3 env_cfg_type=$4 action_type=$5 seed=$6
policy_gpu_id=$7 verne_env=$8 port=$9 host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
# XPolicyLab loads the Model of the policy folder with this name.
policy_name="$(basename "${SCRIPT_DIR}")"

exec env CUDA_VISIBLE_DEVICES="${policy_gpu_id}" PYTHONUNBUFFERED=1 \
    PYTHONPATH="${BENCH_ROOT}:${PYTHONPATH:-}" \
    "${verne_env}/bin/python" -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" \
    --overrides port="${port}" host="${host}" policy_name="${policy_name}" \
        bench_name="${bench_name}" task_name="${task_name}" env_cfg_type="${env_cfg_type}" \
        seed="${seed}" action_type="${action_type}" checkpoint="${checkpoint}"
