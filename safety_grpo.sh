export OPENAI_BASE_URL=https://api2.aigcbest.top/v1
export OPENAI_SAFEGUARD_BASE_URL=http://localhost:8848/v1
export OPENAI_API_KEY=sk-lyUkMdrohSuN3zFWjbUnJpeIowI17fRe9XqpazIBJ8HkFsxd
export WANDB_API_KEY=09f50e022adfb719d63d3e9df0fb0644c2ba3670

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

# export NCCL_P2P_DISABLE=1

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

echo "GPUS_PER_NODE" $GPUS_PER_NODE;

# Jiawei's notes for 4xA100 PCIe (@Yifeng):
# - Becasue of PCIe, prefer gradient checkpointing over offloading
# - If offloading, prefer optimizer offloading (zero1) over param offloading
# - The code execution concurrency is $TOTAL_SAMPLES - nice to make it larger than $(nproc) to maximize CPUs
# - Try to make the #steps as long as possible: e.g., increasing epochs / reducing batches...
# - Set save_freq to a large number as I guess Colossus has little space left
# - If you are short of VRAM, consider removing reference policy. To do so, you need to go to
#    main_ppo.py:main_task - and comment "Role.RefPolicy..." in "role_worker_mapping = ".

# MAIN CONFIG
PROJECT_NAME=safety-r1
EXP_NAME=DeepSeek-R1-Distill-Llama-8B
USE_SCORE=false
VAL_BEFORE_TRAIN=false

MAX_EPOCHS=1
DATASET=safe_rlhf
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
ROLLOUT_N_SAMPLE=16
MICRO_BATCH_PER_GPU=16 # * GPUS_PER_NODE -> GLOBAL_BATCH_SIZE
GLOBAL_BATCH_SIZE=128
GRAD_ACC_STEPS=$(( $GLOBAL_BATCH_SIZE / $MICRO_BATCH_PER_GPU / $GPUS_PER_NODE ))
ROLLOUT_N_QUERY=$(( $GLOBAL_BATCH_SIZE /  $ROLLOUT_N_SAMPLE ))

echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "GRAD_ACC_STEPS: $GRAD_ACC_STEPS"
echo "ROLLOUT_N_QUERY: $ROLLOUT_N_QUERY"
echo "EXP_NAME: $EXP_NAME"

JUDGE_MODEL=meta-llama/Llama-Guard-3-8B
SCORE_MODEL=PKU-Alignment/beaver-7b-v3.0-cost

# assert ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE % GLOBAL_BATCH_SIZE == 0
TOTAL_SAMPLES=$(( $ROLLOUT_N_QUERY * $ROLLOUT_N_SAMPLE ))
if (( $TOTAL_SAMPLES % $GLOBAL_BATCH_SIZE != 0 )); then
    echo "Error: (ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE) must be divisible by GLOBAL_BATCH_SIZE."
    echo "Currently, ${TOTAL_SAMPLES} is not divisible by ${GLOBAL_BATCH_SIZE}."
    exit 1
else
    echo "Assertion passed: ${TOTAL_SAMPLES} is divisible by ${GLOBAL_BATCH_SIZE}."
fi

# export VLLM_ATTENTION_BACKEND=XFORMERS

USE_DYNAMIC_BSZ=true
SP_SIZE=1

VAL_DATA_DIR=./validation/${PROJECT_NAME}-${EXP_NAME}
mkdir -p ${VAL_DATA_DIR}

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/$DATASET/train_10000.parquet \
    data.val_files=data/$DATASET/test_10000.parquet \
    data.train_batch_size=$ROLLOUT_N_QUERY \
    data.val_batch_size=$(( 16 * $ROLLOUT_N_QUERY )) \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$GLOBAL_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS_PER_NODE \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.nnodes=1 \
    trainer.default_local_dir=./checkpoints/${PROJECT_NAME}-${EXP_NAME} \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=64 \
    trainer.test_freq=16 \
    trainer.total_epochs=$MAX_EPOCHS \
    trainer.log_val_generations=-1 \
    trainer.validation_data_dir=$VAL_DATA_DIR \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    reward_model.reward_manager=safety \
    reward_model.enable=$USE_SCORE \
    reward_model.micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    reward_model.model.path=$SCORE_MODEL \
    reward_model.model.use_remove_padding=false \
    grm.enable=false \
    grm.model.path=$JUDGE_MODEL \
    grm.rollout.name=hf \
    grm.micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    grm.rollout.prompt_length=32768 \
    grm.rollout.response_length=128 2>&1 | tee logs/${PROJECT_NAME}-${EXP_NAME}.log
