# Qwen3-VL-30B-A3B-Instruct GRPO 强化训练计划（4 卡 NPU + LoRA）

## 硬件配置

- 芯片：Ascend 910B3（Atlas 200T A2）
- 显存：64GB / 卡
- 使用卡号：NPU 4、5、6、7（后 4 张）
- 总显存：256GB

## 显存估算（LoRA 方案）

| 组件 | 显存/卡 |
|------|--------|
| 基础模型权重（FSDP 分片，4 卡） | ~15GB |
| LoRA 优化器状态（极小） | < 1GB |
| vLLM 推理引擎（TP=4） | ~15GB |
| 激活值 + KV 缓存 | ~20GB |
| **合计** | **~51GB ✅** |

---

## 一、前提条件

在训练机上确认以下软件已安装：

| 软件 | 要求版本 |
|------|---------|
| Python | >= 3.10, < 3.12 |
| CANN | == 8.5.0 |
| torch | == 2.8.0 |
| torch_npu | == 2.8.0 |
| torchvision | == 0.22.1 |
| triton-ascend | == 3.2.0 |
| transformers | == 4.57.6 |
| vllm | v0.13.0 |
| vllm-ascend | releases/v0.13.0 |

**不支持**：flash_attn、liger-kernel（无需安装）

---

## 二、训练机环境安装

```bash
# 1. 激活 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2. 安装 verl
cd /your/path/to/verl
pip install -r requirements-npu.txt
pip install -v -e .
```

---

## 三、训练脚本（4 卡 LoRA 版，使用 NPU 4-7）

将以下内容保存为 `run_qwen3_vl_30b_4npu_lora.sh`，**填写 ① ② 变量**后运行。

```bash
set -x

# ============================================================
# 指定使用后 4 张卡（NPU 4、5、6、7）
# ============================================================
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export USE_OPTIMIZED_MODEL=0
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================
# ① 模型路径（填写 Qwen3-VL-30B-A3B-Instruct 的实际路径）
# ============================================================
MODEL_PATH=${MODEL_PATH:-"/home/l30002999/Qwen3-VL-30B-A3B-Instruct"}

# ============================================================
# ② checkpoint 保存路径
# ============================================================
CKPTS_DIR=${CKPTS_DIR:-"/data/nfs/l30002999/checkpoints/qwen3vl_30b_grpo"}

# ============================================================
# 数据集路径（已确认）
# ============================================================
TRAIN_FILE=/data/nfs/trainning/datasets/performance/geo3k/train.parquet
TEST_FILE=/data/nfs/trainning/datasets/performance/geo3k/test.parquet

project_name='GRPO-Qwen3_vl'
exp_name='GRPO-Qwen3_vl-30B-4npu-lora'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=true \
    algorithm.rollout_correction.rollout_rs=token_k1 \
    algorithm.rollout_correction.rollout_rs_threshold=0.6_1.6 \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15
```

---

## 四、运行方式

```bash
MODEL_PATH=/实际/模型/路径 \
CKPTS_DIR=/实际/checkpoint/路径 \
bash run_qwen3_vl_30b_4npu_lora.sh
```

---

## 五、LoRA 配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `lora_rank` | 16 | LoRA 秩，越大效果越好但越占显存 |
| `lora_alpha` | 32 | 缩放系数，通常设为 rank 的 2 倍 |
| `target_modules` | all-linear | 对所有线性层加 LoRA |
| `exclude_modules` | `.*visual.*` | 排除视觉编码器，只训练语言模型部分 |
| `param_offload` | False | LoRA 后优化器极小，无需卸载 |
| `optimizer_offload` | False | 同上，保持速度 |
| `ref.param_offload` | True | Reference 模型卸载到 CPU |

---

## 六、OOM 应急处理

如果出现显存不足，按顺序尝试：

```bash
# 第一步：减小每卡 micro batch
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

# 第二步：减少每题生成数量
actor_rollout_ref.rollout.n=3

# 第三步：缩短响应长度
data.max_response_length=1024

# 第四步：降低 vLLM 显存占比
actor_rollout_ref.rollout.gpu_memory_utilization=0.75
```

---

## 七、观察训练是否正常

```
train/reward_score  → 应逐步上升（说明模型在学习）
train/kl            → 保持在 0.01 ~ 0.3 之间
```