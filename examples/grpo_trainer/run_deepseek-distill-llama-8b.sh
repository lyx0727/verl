export CUDA_VISIBLE_DEVICES=4,5,6,7
export RAY_TMPDIR=$HOME/lyx/.tmp
export WANDB_API_KEY=09f50e022adfb719d63d3e9df0fb0644c2ba3670


N_GPUS=4
MODEL=$HOME/Models/DeepSeek-R1-Distill-Llama-8B
PROJECT_NAME=rl_grpo_deepseek-r1-distill-llama-8b
EXP_NAME=gsm8k


micro_batch_size=8
mini_batch_size=$((16 * $N_GPUS * $micro_batch_size))
train_batch_size=$mini_batch_size

HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.resume_mode='resume_path' \
    trainer.resume_from_path='/home/mouyutao/lyx/verl/checkpoints/rl_grpo_deepseek-r1-distill-llama-8b/gsm8k/global_step_10' \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=-1 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15;