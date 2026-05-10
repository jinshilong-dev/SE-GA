import os
import gc
import warnings
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from trl import SFTConfig, SFTTrainer

from .config import TrainConfig
from .data import load_grounding_sft_dataset, load_planning_sft_dataset

warnings.filterwarnings("ignore", category=UserWarning)


def _setup_env(env_cfg):
    if env_cfg.SWANLAB_PROJECT:
        os.environ["SWANLAB_PROJECT"] = env_cfg.SWANLAB_PROJECT


def _load_model_and_tokenizer(model_cfg, compute_dtype_override=None):
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    dtype = getattr(torch, compute_dtype_override or model_cfg.compute_dtype)

    tokenizer = AutoProcessor.from_pretrained(model_cfg.model_id, fix_mistral_regex=True)
    processor = AutoProcessor.from_pretrained(
        model_cfg.model_id,
        fix_mistral_regex=True,
        min_pixels=model_cfg.min_pixels,
        max_pixels=model_cfg.max_pixels,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg.model_id,
        device_map=device_map if device_map else "auto",
        dtype=dtype,
        attn_implementation=model_cfg.attn_implementation,
    )

    return model, tokenizer, processor


def _cleanup(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def run_grounding_sft(cfg: TrainConfig):
    _setup_env(cfg.env)

    model_cfg = cfg.model
    model_cfg.model_id = cfg.grounding_sft.model_id
    model, tokenizer, processor = _load_model_and_tokenizer(model_cfg)
    dataset_train = load_grounding_sft_dataset(cfg.grounding_sft)

    sft_cfg = cfg.grounding_sft
    training_args = SFTConfig(
        learning_rate=sft_cfg.learning_rate,
        adam_beta1=sft_cfg.adam_beta1,
        adam_beta2=sft_cfg.adam_beta2,
        weight_decay=sft_cfg.weight_decay,
        warmup_ratio=sft_cfg.warmup_ratio,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        optim=sft_cfg.optim,
        logging_steps=sft_cfg.logging_steps,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        num_train_epochs=sft_cfg.num_train_epochs,
        save_steps=sft_cfg.save_steps,
        max_grad_norm=sft_cfg.max_grad_norm,
        report_to=sft_cfg.report_to,
        run_name=sft_cfg.run_name,
        output_dir=sft_cfg.output_dir_check,
        disable_tqdm=False,
    )

    if sft_cfg.enable_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
    )

    print("=" * 60)
    print("Grounding SFT: Starting training...")
    print("=" * 60)
    trainer.train()
    trainer.save_model(sft_cfg.output_dir_final)
    print(f"Grounding SFT: Model saved to {sft_cfg.output_dir_final}")

    _cleanup(model, tokenizer)


def run_planning_sft(cfg: TrainConfig):
    _setup_env(cfg.env)

    model_cfg = cfg.model
    model_cfg.model_id = cfg.planning_sft.model_id
    model, tokenizer, processor = _load_model_and_tokenizer(model_cfg)
    dataset_train, dataset_eval = load_planning_sft_dataset(cfg.planning_sft)

    sft_cfg = cfg.planning_sft
    training_args = SFTConfig(
        learning_rate=sft_cfg.learning_rate,
        adam_beta1=sft_cfg.adam_beta1,
        adam_beta2=sft_cfg.adam_beta2,
        weight_decay=sft_cfg.weight_decay,
        warmup_ratio=sft_cfg.warmup_ratio,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        optim=sft_cfg.optim,
        logging_steps=sft_cfg.logging_steps,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        num_train_epochs=sft_cfg.num_train_epochs,
        save_steps=sft_cfg.save_steps,
        max_grad_norm=sft_cfg.max_grad_norm,
        report_to=sft_cfg.report_to,
        run_name=sft_cfg.run_name,
        output_dir=sft_cfg.output_dir_check,
        disable_tqdm=False,
        eval_steps=sft_cfg.eval_steps,
        eval_strategy=sft_cfg.eval_strategy,
        do_eval=sft_cfg.do_eval,
    )

    if sft_cfg.enable_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
    )

    print("=" * 60)
    print("Planning SFT: Starting training...")
    print("=" * 60)
    trainer.train()
    trainer.save_model(sft_cfg.output_dir_final)
    print(f"Planning SFT: Model saved to {sft_cfg.output_dir_final}")

    _cleanup(model, tokenizer)


def run_sft(cfg: TrainConfig):
    print("\n" + "#" * 60)
    print("# SFT Pipeline: Grounding + Planning")
    print("#" * 60 + "\n")

    run_grounding_sft(cfg)

    print("\n" + "-" * 60)
    print("Grounding SFT done. Starting Planning SFT...")
    print("-" * 60 + "\n")

    cfg.planning_sft.model_id = cfg.grounding_sft.output_dir_final
    run_planning_sft(cfg)

    print("\n" + "#" * 60)
    print("# SFT Pipeline completed!")
    print(f"# Grounding model: {cfg.grounding_sft.output_dir_final}")
    print(f"# Planning model: {cfg.planning_sft.output_dir_final}")
    print("#" * 60)
