#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SGLang benchmark using sglang.bench_serving
# Expects: endpoint isl osl concurrencies [req_rate]
#
# Environment variables for profiling:
#   PROFILE_TYPE: "nsys" or "torch" to enable profiling
#   PROFILE_PREFILL_IPS: Comma-separated list of prefill worker IPs
#   PROFILE_DECODE_IPS: Comma-separated list of decode worker IPs
#   PROFILE_PREFILL_START_STEP / PROFILE_PREFILL_STOP_STEP: Step range for prefill
#   PROFILE_DECODE_START_STEP / PROFILE_DECODE_STOP_STEP: Step range for decode

set -e

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

MODEL_NAME="${BENCH_MODEL_NAME:-deepseek-ai/DeepSeek-R1}"
TOKENIZER_PATH="${BENCH_TOKENIZER:-/model}"

echo "SGLang-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}; tokenizer=${TOKENIZER_PATH}"

# Profiling shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/profiling.sh
source "${SCRIPT_DIR}/../lib/profiling.sh"
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Parse concurrency list
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

resolve_profile_target_concurrency() {
    local target="${PROFILE_TARGET_CONCURRENCY:-max}"
    local count="${#CONCURRENCY_LIST[@]}"
    if [[ "${count}" -eq 0 ]]; then
        echo ""
        return 0
    fi

    case "${target}" in
        first)
            echo "${CONCURRENCY_LIST[0]}"
            ;;
        last)
            echo "${CONCURRENCY_LIST[$((count - 1))]}"
            ;;
        ""|max)
            local max="${CONCURRENCY_LIST[0]}"
            local concurrency
            for concurrency in "${CONCURRENCY_LIST[@]}"; do
                if [[ "${concurrency}" -gt "${max}" ]]; then
                    max="${concurrency}"
                fi
            done
            echo "${max}"
            ;;
        *)
            echo "${target}"
            ;;
    esac
}

if [[ "${PROFILE_TRIGGER:-before-benchmark}" == "benchmark-measurement" ]]; then
    PROFILE_RESOLVED_TARGET_CONCURRENCY="$(resolve_profile_target_concurrency)"
fi

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"

if [[ "${PROFILE_TRIGGER:-before-benchmark}" == "before-benchmark" ]]; then
    start_all_profiling
elif [[ "${PROFILE_TRIGGER}" == "benchmark-measurement" ]]; then
    echo "Profiling will start for measured concurrency: ${PROFILE_RESOLVED_TARGET_CONCURRENCY}"
else
    echo "Warning: unsupported PROFILE_TRIGGER='${PROFILE_TRIGGER}', profiling will not start"
fi

profile_triggered=0

# Run benchmark for each concurrency level
for concurrency in "${CONCURRENCY_LIST[@]}"; do
    echo "Running benchmark with concurrency: $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"

    profile_this_concurrency=0
    if [[ "${PROFILE_TRIGGER:-before-benchmark}" == "benchmark-measurement" && "${concurrency}" == "${PROFILE_RESOLVED_TARGET_CONCURRENCY}" ]]; then
        PROFILE_CURRENT_CONCURRENCY="${concurrency}"
        profile_this_concurrency=1
        profile_triggered=1
        start_all_profiling
    fi

    set -x
    python3 -m sglang.bench_serving \
        --backend sglang-oai \
        --model "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --host "${HOST}" --port "${PORT}" \
        --dataset-name random \
        --max-concurrency "${concurrency}" \
        --num-prompts 128 \
        --random-input-len "${ISL}" \
        --random-output-len "${OSL}" \
        --random-range-ratio 1 \
        --request-rate "${REQ_RATE}" \
        --warmup-requests 0
    set +x

    if [[ "${profile_this_concurrency}" == "1" ]]; then
        stop_all_profiling
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

if [[ "${PROFILE_TRIGGER:-before-benchmark}" == "benchmark-measurement" && "${profile_triggered}" != "1" ]]; then
    echo "Warning: target profiling concurrency '${PROFILE_RESOLVED_TARGET_CONCURRENCY}' was not found"
fi

stop_all_profiling

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "SGLang-Bench completed"
