#!/bin/bash


model="meta-llama/Llama-3.2-1B"

# vllm serve meta-llama/Llama-3.2-1B --no-enable-prefix-caching --enforce-eager --load-format=dummy --disable-log-requests

python benchmark_serving.py --dataset-name=sharegpt --dataset-path=ShareGPT_V3_unfiltered_cleaned_split.json --model=$model
