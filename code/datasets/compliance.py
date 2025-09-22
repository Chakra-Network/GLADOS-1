import torch
from torch.utils.data import Dataset
import random
from code.consts import COMPLIANCE_PROMPT


ACTION_TYPES = [
    "click",
    "left_double",
    "right_single",
]


class ComplianceSampleDataset(Dataset):
    """Custom Dataset for compliance samples with proper batching."""

    def __init__(self, n, processor):
        self.n = n
        self.indices = list(range(n))
        self.processor = processor

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        input_ids, attention_mask, labels = (
            convert_compliance_index_to_training_message(
                idx,
                None,
                self.processor,
            )
        )
        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


def _get_action_content_and_instruction(action_type):
    x, y = random.randint(0, 1000), random.randint(0, 1000)
    match action_type:
        case "click":
            return (
                f"click(start_box='<|box_start|>({x}, {y})<|box_end|>')",
                f"Please click on the element at {x}, {y}",
            )
        case "left_double":
            return (
                f"left_double(start_box='<|box_start|>({x}, {y})<|box_end|>')",
                f"Please double click on the element at {x}, {y}",
            )
        case "right_single":
            return (
                f"right_single(start_box='<|box_start|>({x}, {y})<|box_end|>')",
                f"Please right click on the element at {x}, {y}",
            )
        case _:
            raise ValueError(f"Unknown action type {action_type}")


def generate_conversation(idx):
    action_type = ACTION_TYPES[idx % len(ACTION_TYPES)]
    action_content, instruction = _get_action_content_and_instruction(action_type)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": COMPLIANCE_PROMPT.format(instruction=instruction),
                },
            ],
        },
        {
            "role": "assistant",
            "content": f"Action: {action_content}",
        },
    ]


def convert_compliance_index_to_training_message(index, _, processor):
    conversation = generate_conversation(index)

    new_sample = []
    for frame in conversation:
        # get all fields except loss_mask
        new_frame = {k: v for k, v in frame.items() if k != "loss_mask"}
        new_sample.append(new_frame)

    sample_text = [new_sample[0]]
    target = new_sample[1]["content"]

    text_input = processor.apply_chat_template(
        sample_text, tokenize=False, add_generation_prompt=True
    )

    input_tokens = processor(
        text=text_input,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")
    target_tokens = processor.tokenizer(
        target,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,  # Don't add extra special tokens
    ).to("cuda")

    input_ids = torch.cat(
        [input_tokens["input_ids"], target_tokens["input_ids"]], dim=1
    )

    attention_mask = torch.cat(
        [input_tokens["attention_mask"], target_tokens["attention_mask"]], dim=1
    )

    labels = torch.cat(
        [
            torch.full_like(input_tokens["input_ids"], -100),  # Ignore input in loss
            target_tokens["input_ids"],  # Learn to predict target
        ],
        dim=1,
    )

    return input_ids, attention_mask, labels
