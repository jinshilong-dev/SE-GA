import os
import warnings
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig as TRL_GRPOConfig, GRPOTrainer

from .config import TrainConfig
from .data import load_grpo_dataset
from .rewards import format_reward_func, thinking_reward_func, accuracy_reward_type, accuracy_reward_action

warnings.filterwarnings("ignore", category=UserWarning)


def run_grpo(cfg: TrainConfig):
    model_cfg = cfg.model
    model_cfg.model_id = cfg.grpo.model_id
    grpo_cfg = cfg.grpo
    env_cfg = cfg.env

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if env_cfg.SWANLAB_PROJECT:
        os.environ["SWANLAB_PROJECT"] = env_cfg.SWANLAB_PROJECT

    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    compute_dtype = getattr(torch, grpo_cfg.compute_dtype)

    tokenizer = AutoProcessor.from_pretrained(model_cfg.model_id, fix_mistral_regex=True)
    processor = AutoProcessor.from_pretrained(
        model_cfg.model_id,
        fix_mistral_regex=True,
        min_pixels=model_cfg.min_pixels,
        max_pixels=model_cfg.max_pixels,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg.model_id,
        device_map=device_map,
        dtype=compute_dtype,
        attn_implementation=model_cfg.attn_implementation,
    )

    dataset_train, dataset_eval = load_grpo_dataset(grpo_cfg)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    peft_config = None
    if grpo_cfg.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=grpo_cfg.lora_r,
            lora_alpha=grpo_cfg.lora_alpha,
            target_modules=grpo_cfg.lora_target_modules,
            bias=grpo_cfg.lora_bias,
        )

    rollout_n = (grpo_cfg.per_device_train_batch_size / grpo_cfg.num_generations) * grpo_cfg.gradient_accumulation_steps * grpo_cfg.gpu_num

    training_args = TRL_GRPOConfig(
        use_vllm=grpo_cfg.use_vllm,
        vllm_mode=grpo_cfg.vllm_mode,
        learning_rate=grpo_cfg.learning_rate,
        adam_beta1=grpo_cfg.adam_beta1,
        adam_beta2=grpo_cfg.adam_beta2,
        weight_decay=grpo_cfg.weight_decay,
        lr_scheduler_type=grpo_cfg.lr_scheduler_type,
        optim=grpo_cfg.optim,
        beta=grpo_cfg.beta,
        logging_steps=grpo_cfg.logging_steps,
        bf16=grpo_cfg.bf16,
        fp16=grpo_cfg.fp16,
        per_device_train_batch_size=grpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_cfg.gradient_accumulation_steps,
        num_generations=grpo_cfg.num_generations,
        max_prompt_length=grpo_cfg.max_prompt_length,
        max_completion_length=grpo_cfg.max_completion_length,
        num_train_epochs=grpo_cfg.num_train_epochs,
        save_steps=grpo_cfg.save_steps,
        max_grad_norm=grpo_cfg.max_grad_norm,
        report_to=grpo_cfg.report_to,
        run_name=grpo_cfg.run_name,
        output_dir=grpo_cfg.output_dir_check,
        epsilon_high=grpo_cfg.epsilon_high,
        deepspeed=grpo_cfg.deepspeed_config,
        gradient_checkpointing=grpo_cfg.gradient_checkpointing,
        disable_tqdm=False,
        eval_steps=grpo_cfg.eval_steps,
        eval_strategy=grpo_cfg.eval_strategy,
        do_eval=grpo_cfg.do_eval,
    )

    if grpo_cfg.use_lora:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    reward_funcs = [format_reward_func, thinking_reward_func, accuracy_reward_type, accuracy_reward_action]
    num_train_dataset = int(grpo_cfg.num_train_epochs * len(dataset_train) / rollout_n)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        peft_config=None,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        num_dataset=num_train_dataset,
    )

    print("=" * 60)
    print("GRPO RL: Starting training...")
    print("=" * 60)
    trainer.train()
    trainer.save_model(grpo_cfg.output_dir_final)
    print(f"GRPO RL: Model saved to {grpo_cfg.output_dir_final}")
