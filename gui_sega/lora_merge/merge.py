import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoProcessor.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


model_name = './model/2_sft_re_full_1600_1'
output = './model/3_rl_1600'
lora = './model/3_rl_re_1600'
apply_lora(model_name, output, lora)

# CUDA_VISIBLE_DEVICES=0 python delete.