#!/bin/bash
# run_models.sh

model_paths=(
    "/home/vilab/projects/video_Reinforcement/small_math_r1/src/r1-v/log/small_math_r1"
)

file_names=(
    "/home/vilab/projects/video_Reinforcement/small_math_r1/src/eval/math-500/eval_math500.json"
)

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval/eval_math_bench.py --model_path "$model" --file_name "$file_name"
done
