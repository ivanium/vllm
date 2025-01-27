#!/bin/bash

MODEL_NAME="meta-llama/Llama-2-7b-hf"

PROJ_ROOT_DIR=$(realpath $(pwd)/../../)
BENCHMARKS_DIR=$(realpath $PROJ_ROOT_DIR/benchmarks)
DAEMON_LOG_FILE="vllm-server.log"

pushd $BENCHMARKS_DIR

if [[ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

vllm serve $MODEL_NAME --disable-log-requests --load-format=dummy > $DAEMON_LOG_FILE 2>&1 &
SERVER_PID=$!

while ! grep -q "Uvicorn running on" $DAEMON_LOG_FILE; do
    sleep 1
done

echo "vLLM server is up and running"

python benchmark_serving.py --backend=vllm --model=$MODEL_NAME --dataset-path=ShareGPT_V3_unfiltered_cleaned_split.json

kill $SERVER_PID
popd  # $BENCHMARKS_DIR