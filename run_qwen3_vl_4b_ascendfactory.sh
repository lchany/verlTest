set -x

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_memory_monitor_refresh_ms=0
export RAY_object_store_memory=10737418240

ascendfactory-cli train /home/l30002999/verl_qwen3_vl-4b_ppo_fsdp.yaml \
    --backend_config.actor_rollout_ref.model.path=/data/nfs/model/Qwen3-VL-4B-Instruct \
    --backend_config.data.train_files=/data/nfs/trainning/datasets/performance/geo3k/train.parquet \
    --backend_config.data.val_files=/data/nfs/trainning/datasets/performance/geo3k/test.parquet \
    --backend_config.trainer.save_freq=50 \
    --backend_config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    --backend_config.actor_rollout_ref.ref.fsdp_config.param_offload=False \
    --backend_config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    --af_output_dir=/data/nfs/l30002999/checkpoints/output/qwen3vl_4b_grpo_2npu \
    --env.MASTER_ADDR=localhost \
    --env.NNODES=1 \
    --env.NODE_RANK=0 \
    | tee /home/l30002999/qwen3_vl-4b_grpo_fsdp_A2_start.log
