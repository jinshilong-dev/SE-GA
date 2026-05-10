import os
import json
import random
from PIL import Image
from datasets import Dataset

from .prompts import SYSTEM_PROMPT_STAGE1, SYSTEM_PROMPT_STAGE2


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _truncate_history(history, max_steps):
    if len(history) > max_steps:
        return history[len(history) - max_steps:]
    return history


def _build_thinking_answer(screen_description, intention, instruction, action_type, action_info):
    answer_dict = {"action_type": action_type, "action_info": action_info}
    answer_json = json.dumps(answer_dict, ensure_ascii=False)
    return (
        "<thinking>"
        + "<analysis>" + screen_description + "</analysis>"
        + "<reasoning>" + intention + "</reasoning>"
        + "<instruction>" + instruction + "</instruction>"
        + "</thinking>\n"
        + "<answer>" + answer_json + "</answer>"
    )


def _open_and_resize_image(image_path, resize):
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize((resize, resize))


def _process_rl_action_info(example):
    action_type = example["action_type"]
    if action_type == "CLICK" and example.get("sam2_bbox") and example["sam2_bbox"] != []:
        action_info = [example["sam2_bbox"][i:i + 2] for i in range(0, len(example["sam2_bbox"]), 2)]
    elif action_type == "LONG_PRESS" and example.get("sam2_bbox") and example["sam2_bbox"] != []:
        action_info = example["action_info"][0]
    else:
        action_info = example["action_info"]
    return action_type, action_info


def load_grounding_sft_dataset(cfg):
    ds = _load_json(cfg.dataset_path)

    def gen():
        for example in ds:
            image_path = os.path.join(cfg.image_dir, example["image_path"])
            task = example["instruction"]
            answer_dict = {"action_type": example["action_type"], "action_info": example["action_info"]}
            answer = json.dumps(answer_dict, ensure_ascii=False)
            img = _open_and_resize_image(image_path, cfg.image_resize)
            yield {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_STAGE1}]},
                    {"role": "user", "content": [
                        {"type": "image_pil", "image": img},
                        {"type": "text", "text": task},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            }

    return Dataset.from_generator(gen)


def load_planning_sft_dataset(cfg):
    data_set = _load_json(cfg.dataset_path)
    data_eval = _load_json(cfg.dataset_eval_path)

    random.seed(42)
    random.shuffle(data_set)
    random.shuffle(data_eval)

    train_ds = data_set[:cfg.train_data_limit]
    eval_ds = data_eval[:cfg.eval_data_limit]

    def gen_train():
        for example in train_ds:
            image_path = os.path.join(cfg.image_dir, example["image_path"])
            task = example["task"]
            screen_description = example["description"]
            history_json = _truncate_history(example["history"], cfg.max_history_steps)
            intention = example["intention"]
            instruction = example["instruction"]
            action_type = example["action_type"]
            action_info = example["action_info"]
            answer = _build_thinking_answer(screen_description, intention, instruction, action_type, action_info)
            history = json.dumps(history_json, ensure_ascii=False)
            img = _open_and_resize_image(image_path, cfg.image_resize)
            yield {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_STAGE2}]},
                    {"role": "user", "content": [
                        {"type": "image_pil", "image": img},
                        {"type": "text", "text": "Your task is: " + task + "\n Here is a brief summary of the previous 10 steps, which led us to the current task stage: " + history},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            }

    def gen_eval():
        for example in eval_ds:
            image_path = os.path.join(cfg.image_dir, example["image_path"])
            task = example["task"]
            screen_description = example["description"]
            history_json = _truncate_history(example["history"], cfg.max_history_steps)
            intention = example["intention"]
            instruction = example["instruction"]
            action_type = example["action_type"]
            action_info = example["action_info"]
            answer = _build_thinking_answer(screen_description, intention, instruction, action_type, action_info)
            history = json.dumps(history_json, ensure_ascii=False)
            img = _open_and_resize_image(image_path, cfg.image_resize)
            yield {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_STAGE2}]},
                    {"role": "user", "content": [
                        {"type": "image_pil", "image": img},
                        {"type": "text", "text": "Your task is: " + task + "\n Here is a brief summary of the previous 10 steps, which led us to the current task stage: " + history},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            }

    dataset_train = Dataset.from_generator(gen_train)
    dataset_eval = Dataset.from_generator(gen_eval)
    return dataset_train, dataset_eval


def load_grpo_dataset(cfg):
    data_test = _load_json(cfg.dataset_test_path)
    ds = _load_json(cfg.dataset_path)

    random.seed(42)
    random.shuffle(ds)
    random.shuffle(data_test)

    train_ds = ds[:cfg.train_data_limit]
    eval_ds = data_test[:cfg.eval_data_limit]

    def gen_train():
        for example in train_ds:
            image_path = os.path.join(cfg.image_dir, example["image_path"])
            task = example["task"]
            screen_description = example["description"]
            history_json = _truncate_history(example["history"], cfg.max_history_steps)
            intention = example["intention"]
            instruction = example["instruction"]
            action_type, action_info = _process_rl_action_info(example)
            answer = _build_thinking_answer(screen_description, intention, instruction, action_type, action_info)
            history = json.dumps(history_json, ensure_ascii=False)
            img = _open_and_resize_image(image_path, cfg.image_resize)
            yield {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_STAGE2}]},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Your task is: " + task + "\n Here is a brief summary of the previous 10 steps, which led us to the current task stage: " + history},
                    ]},
                ],
                "image": img,
                "solution": answer,
            }

    def gen_eval():
        for example in eval_ds:
            image_path = os.path.join(cfg.image_dir_test, example["image_path"])
            task = example["task"]
            screen_description = example["description"]
            history_json = _truncate_history(example["history"], cfg.max_history_steps)
            intention = example["intention"]
            instruction = example["instruction"]
            action_type, action_info = _process_rl_action_info(example)
            answer = _build_thinking_answer(screen_description, intention, instruction, action_type, action_info)
            history = json.dumps(history_json, ensure_ascii=False)
            img = _open_and_resize_image(image_path, cfg.image_resize)
            yield {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_STAGE2}]},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Your task is: " + task + "\n Here is a brief summary of the previous 10 steps, which led us to the current task stage: " + history},
                    ]},
                ],
                "image": img,
                "solution": answer,
            }

    dataset_train = Dataset.from_generator(gen_train)
    dataset_eval = Dataset.from_generator(gen_eval)
    return dataset_train, dataset_eval
