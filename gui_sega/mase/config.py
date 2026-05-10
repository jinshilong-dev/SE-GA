import yaml
from dataclasses import dataclass, field


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelConfig:
    compute_dtype: str = "float16"
    attn_implementation: str = "flash_attention_2"
    min_pixels: int = 65536
    max_pixels: int = 6553600


@dataclass
class EnvConfig:
    SWANLAB_PROJECT: str = "gui-sega"


@dataclass
class GroundingSFTConfig:
    model_id: str = ""
    dataset_path: str = ""
    image_dir: str = ""
    output_dir_check: str = "./checkpoint/grounding"
    output_dir_final: str = "./model/grounding"
    run_name: str = "grounding-sft"
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    save_steps: int = 500
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    report_to: str = "swanlab"
    image_resize: int = 1000
    enable_gradient_checkpointing: bool = True
    use_lora: bool = False


@dataclass
class PlanningSFTConfig:
    model_id: str = ""
    dataset_path: str = ""
    dataset_eval_path: str = ""
    image_dir: str = ""
    output_dir_check: str = "./checkpoint/planning"
    output_dir_final: str = "./model/planning"
    run_name: str = "planning-sft"
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_8bit"
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 2
    save_steps: int = 3000
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    report_to: str = "swanlab"
    eval_steps: int = 200
    eval_strategy: str = "steps"
    do_eval: bool = True
    train_data_limit: int = 1600
    eval_data_limit: int = 160
    image_resize: int = 1000
    enable_gradient_checkpointing: bool = False
    use_lora: bool = False
    max_history_steps: int = 10


@dataclass
class GRPOConfig:
    model_id: str = ""
    dataset_path: str = ""
    dataset_test_path: str = ""
    image_dir: str = ""
    image_dir_test: str = ""
    output_dir_check: str = "./checkpoint/grpo"
    output_dir_final: str = "./model/grpo"
    run_name: str = "grpo-rl"
    compute_dtype: str = "bfloat16"
    deepspeed_config: str = "ds_z3_offload_config.json"
    learning_rate: float = 3e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    lr_scheduler_type: str = "constant"
    optim: str = "adamw_torch_8bit"
    beta: float = 0.001
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    num_generations: int = 16
    num_train_epochs: int = 15
    save_steps: int = 250
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    report_to: str = "swanlab"
    eval_steps: int = 250
    eval_strategy: str = "steps"
    do_eval: bool = True
    max_prompt_length: int = 6144
    max_completion_length: int = 1024
    epsilon_high: float = 0.3
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    train_data_limit: int = 2000
    eval_data_limit: int = 200
    image_resize: int = 1000
    max_history_steps: int = 10
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_target_modules: str = "all-linear"
    lora_bias: str = "none"
    gpu_num: int = 2


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    grounding_sft: GroundingSFTConfig = field(default_factory=GroundingSFTConfig)
    planning_sft: PlanningSFTConfig = field(default_factory=PlanningSFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)


def _dict_to_dataclass(cls, d):
    if d is None:
        return cls()
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k in field_types:
            kwargs[k] = v
    return cls(**kwargs)


def parse_config(config_path: str) -> TrainConfig:
    raw = load_config(config_path)
    return TrainConfig(
        model=_dict_to_dataclass(ModelConfig, raw.get("model")),
        env=_dict_to_dataclass(EnvConfig, raw.get("env")),
        grounding_sft=_dict_to_dataclass(GroundingSFTConfig, raw.get("grounding_sft")),
        planning_sft=_dict_to_dataclass(PlanningSFTConfig, raw.get("planning_sft")),
        grpo=_dict_to_dataclass(GRPOConfig, raw.get("grpo")),
    )
