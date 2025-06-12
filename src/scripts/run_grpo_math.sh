cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=1      # RDMA 없으면 필수
export TORCH_NCCL_BLOCKING_WAIT=1

# ===== 🔍 디버깅용 로그 =====
echo "🚀 [SH] GRPO 실행 시작"
echo "📁 현재 디렉토리: $(pwd)"
echo "🧠 모델 경로 체크: $(pwd)/log/small_math_r1"
if [ ! -d "./log/small_math_r1" ]; then
    echo "❌ 모델 디렉토리가 존재하지 않습니다: ./log/small_math_r1"
    exit 1
else
    echo "✅ 모델 디렉토리 존재 확인 완료"
fi


echo "🏁 torchrun 실행 시작"

# ==========================
CUDA_VISIBLE_DEVICES=6,7,8 \
torchrun --nproc_per_node=3 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12400" \
    src/open_r1/grpo_math2.py \
    --output_dir "./log/RF_small_math_r1_5" \
    --model_name_or_path './log/small_math_r1' \
    --dataset_name "data/Math/final_math_r1_100.json" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --fp16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal false \
    --len_control true \
    --attn_implementation "flash_attention_2" \
    --max_pixels 401408 \
    --num_train_epochs 30 \
    --run_name rf_small-math-r1 \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

# CUDA_VISIBLE_DEVICES=6 python src/open_r1/grpo_math2.py \
#     --output_dir "./log/RF_small_math_r1_4" \
#     --model_name_or_path './log/small_math_r1' \
#     --dataset_name "data/Math/final_math_r1_1000.json" \
#     --deepspeed local_scripts/zero3.json \
#     --max_prompt_length 16384 \
#     --max_completion_length 768 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 1e-6 \
#     --lr_scheduler_type "cosine" \
#     --weight_decay 0.01 \
#     --fp16 \
#     --logging_steps 1 \
#     --gradient_checkpointing true \
#     --temporal false \
#     --len_control true \
#     --attn_implementation "flash_attention_2" \
#     --max_pixels 401408 \
#     --num_train_epochs 30 \
#     --run_name rf_small-math-r1 \
#     --save_steps 100 \
#     --beta 0.04 \
#     --max_grad_norm 5 \
#     --save_only_model false \
#     --num_generations 8


