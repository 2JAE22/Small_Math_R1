cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"


CUDA_VISIBLE_DEVICES=6,7,8 torchrun --nproc_per_node="10" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --output_dir "./log/small_math_r1" \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "data/Math/mathR1-cot.json" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-3B-Video-cot-sft \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \