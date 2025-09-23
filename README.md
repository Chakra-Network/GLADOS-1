<div align="center">
<p align="center">
  <h1>GLADOS-1</h1>
</p>
</div>


<img width="1500" height="675" alt="glados-1" src="https://github.com/user-attachments/assets/135265ee-1fa8-4f88-9a6d-1d8cbdc9f5f9" />
<div align="center">
<p>
        📕 <a href="https://www.chakra.dev/research/glados-1-compute-use-model-crowdsourced-trajectories">Release Blog</a>&nbsp&nbsp | &nbsp&nbsp 🤗 <a href="https://huggingface.co/chakra-labs/GLADOS-1">Hugging Face Model</a>&nbsp&nbsp 
        | &nbsp&nbsp 🔧 <a href="https://github.com/bytedance/UI-TARS/blob/main/README_deploy.md">Deployment (via UI-TARS)</a> &nbsp&nbsp  | &nbsp&nbsp
🖥️ <a href="https://github.com/bytedance/UI-TARS-desktop">Running on your own computer (via UI-TARS Desktop)</a>&nbsp&nbsp
</p>
</div>

**GLADOS-1** is the first computer-use (CUA) model post-trained using collective, crowd-sourced trajectories via the [PANGO dataset](https://huggingface.co/datasets/chakra-labs/pango-sample).


## Overview

Heavily inspired by the [Qwen-2VL-Finetune repository](https://github.com/2U1/Qwen2-VL-Finetune), this project provides a framework for training vision-language models on GUI interaction data. While this code represents sample code for post-training [UI-TARS-7B-SFT](https://huggingface.co/ByteDance-Seed/UI-TARS-7B-SFT) via ByteDance Seed, it can be trivially updated for any model based on the Qwen2-VL architecture. 

The [PANGO](https://huggingface.co/datasets/chakra-labs/pango) (**P**roductivity **A**pplications with **N**atural **G**UI **O**bservations and trajectories) dataset contains real user interactions with web interfaces, converted into training conversations for multimodal models.

## Dataset Structure

Each session in the PANGO dataset contains:

- **Screenshots**: GUI state images at different timestamps
- **Actions**: User interactions (clicks, drags, typing, etc.)
- **Metadata**: Session IDs, timestamps, and other inputs

### Action Types

The dataset supports various GUI interaction types:

**Supported Actions:**

- `click` - Single left mouse clicks
- `left_double` - Double left mouse clicks
- `right_single` - Right mouse clicks
- `drag` - Mouse drag operations (converted from drag_start/drag_end pairs)
- `key_press` - Keyboard key presses
- `input` - Text input actions
- `scroll` - Scroll wheel actions

**Ignored Actions:**

- `mouseover_start` / `mouseover_end` - Mouse hover events
- `drag_start` / `drag_end` - Individual drag events (converted to single `drag`)

## Converters

Converters transform raw Pango data into training conversations. Each converter implements a specific training purpose:

### 1. SimpleGroundingConverter

- **Input**: Single screenshot and instruction
- **Output**: Action prediction
- **Use Case**: Instruction-following GUI automation

### 2. StateTransitionConverter

- **Input**: Before and after screenshots
- **Output**: Action prediction
- **Use Case**: Reverse engineering user interactions

### 3. MultiTurnConverter (Beta)

- **Input**: Conversational history containing screenshots and actions
- **Output**: Action prediction
- **Use Case**: Multi-turn conversation training

## Getting Started

### Installation

```bash
# Install uv package manager
brew install uv

# Install dependencies
make install
```

### Training

```bash
# Train with grounding dataset
make train

# Train with state transition dataset
make train_state_transition
```

### Storage Requirements

During setup, the script `image_downloader` script will download all images to the `STORAGE_DIR` directory. The estimated storage requirements for the `pango-sample` dataset is 15 GB, and 265 GB for the `pango` full dataset. Note, the image downloader script has a hardcoded buffer of 50GB, adjust and rebuild if this is an issue.

## Extending Converters

To create a new converter:

1. **Inherit from BasePangoConverter**:

```python
from code.converters.base_pango_converter import BasePangoConverter

class MyConverter(BasePangoConverter):
    def __init__(self, dataset_path: str, prompt: str, **kwargs):
        super().__init__(dataset_path, actions_to_ignore=[...], **kwargs)
        self.prompt = prompt
```

2. **Implement required methods**:

```python
def generate_conversation(self, *args, **kwargs) -> list:
    """Convert actions to training conversation format"""
    # Return list of conversation frames with:
    # - role: "user" or "assistant"
    # - content: text/image content
    # - loss_mask: 0 (ignore) or 1 (train on)
    pass

def generate_indices(self, n: int, pct_train: float) -> tuple[list, list]:
    """Generate train/test indices for the dataset"""
    # Return (train_indices, test_indices)
    pass
```

3. **Add action handling** (as needed):

```python
def _handle_custom_action(self, action: dict, original_dims, scaled_dims):
    """Handle new action types"""
    # Convert pango action to model action format
    return action_content
```

4. **Create corresponding dataset class**:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, indices: list[int], converter: MyConverter, processor):
        self.indices = indices
        self.converter = converter
        self.processor = processor

    def __getitem__(self, idx):
        # Convert index to training sample
        return {"input_ids": ..., "attention_mask": ..., "labels": ...}
```

### Key Implementation Notes

- **Coordinate Scaling**: Actions use standardized coordinates (0-1000 range)
- **Image Processing**: Screenshots are resized and processed using `fetch_image` from `qwen-vl-utils`
- **Error Handling**: Use `_handle_error()` and `_handle_malformatted_action()` for graceful failures
- **Lazy Loading**: Images are loaded on demand during training by the `__getitem__` method on the dataset class

## Project Structure

```
code/
├── converters/         # Data conversion logic
├── datasets/           # PyTorch dataset implementations
├── training/           # Training scripts and utilities
└── train.py            # Main training entry point
└── utils.py            # Utility functions
└── consts.py           # Constants
└── exceptions.py       # Custom exceptions
└── tests/              # Test files
```


## Citation

```tex
@misc{chakralabs2025glados-1,
  author = {Chakra Labs},
  title = {GLADOS-1},
  url = {https://github.com/Chakra-Network/GLADOS-1},
  year = {2025}
}
```
