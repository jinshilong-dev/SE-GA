## Dataset
Due to the large size of the dataset, please go to Hugging Face to download it：[SE-GA-dataset](https://huggingface.co/datasets/waterphd/SE-GA-dataset)


## Dataset Format

### Grounding SFT Data

```json
[
  {
    "image_path": "image_001.png",
    "instruction": "Open the Settings app",
    "action_type": "CLICK",
    "action_info": [500, 300]
  }
]
```

### Planning SFT & GRPO Data

```json
[
  {
    "image_path": "screen_001.png",
    "task": "Send a message to John",
    "description": "Home screen with app icons visible",
    "history": [{"action_type": "CLICK", "action_info": [500, 300]}],
    "intention": "Need to open the Messages app",
    "instruction": "Tap the Messages app icon",
    "action_type": "CLICK",
    "action_info": [500, 300],
    "sam2_bbox": [480, 280, 520, 320]
  }
]
```

**Key fields:**
- `sam2_bbox`: For CLICK actions, it's converted to `[x1,y1], [x2,y2]` format for reward computation.
- `history`: List of past actions (truncated to last `max_history_steps` entries).
- `description` / `intention` / `instruction`: Used to construct the `<thinking>` block.