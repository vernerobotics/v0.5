#!/bin/bash
# Run one RoboTwin task with Verne V0.5: start the policy server, then the sim.
#
#   bash XPolicyLab/policy/Verne/eval.sh RoboTwin <task> <checkpoint> aloha_agilex joint \
#       <seed> <policy_gpu> <sim_gpu> <verne_env> <robotwin_conda_env>
#
# <verne_env> is the folder of the Python environment that has Verne V0.5 installed.
set -euo pipefail

bench_name=$1 task_name=$2 checkpoint=$3 env_cfg_type=$4 action_type=$5 seed=$6
policy_gpu_id=$7 env_gpu_id=$8 verne_env=$9 robotwin_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

port=$(bash "${UTILS_DIR}/get_free_port.sh")
bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" "${bench_name}" "${task_name}" "${checkpoint}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${policy_gpu_id}" "${verne_env}" "${port}" &
server=$!
trap 'kill "${server}" 2>/dev/null || true' EXIT
bash "${UTILS_DIR}/wait_for_policy_server.sh" localhost "${port}" "${server}" "Policy server" 1200

CUDA_VISIBLE_DEVICES="${env_gpu_id}" bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" "${SCRIPT_DIR}/deploy.yml" "${robotwin_env}" "${port}" "${bench_name}" \
    "${task_name}" "${env_cfg_type}" "$(basename "${SCRIPT_DIR}")" \
    "ckpt_name=$(basename "${checkpoint}"),action_type=${action_type}" "${BENCH_ROOT}" \
    "${seed}" "${env_gpu_id}" localhost
