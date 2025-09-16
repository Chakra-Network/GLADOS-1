import torch
from torch.utils.data import Dataset

from ..converters.simple_grounding_converter import SimpleGroundingConverter
from ..exceptions import ActionConversionError


class GroundingSampleDataset(Dataset):
    """Custom Dataset for grounding samples with proper batching."""

    def __init__(
        self, indices: list[int], converter: SimpleGroundingConverter, processor
    ):
        self.indices = indices
        self.converter = converter
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        action_index = self.indices[idx]
        input_ids, attention_mask, labels = convert_grounding_index_to_training_message(
            action_index, self.converter, self.processor
        )
        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


def convert_grounding_index_to_training_message(
    index: int, converter: SimpleGroundingConverter, processor
):
    """
    Takes a sample (with training masks) and convert it to a tuple with (input_ids, attention_mask, labels)
    """
    conversation = converter.generate_conversation(converter.actions[index])

    if not conversation:
        raise ActionConversionError("No conversation generated")

    new_sample = []
    for frame in conversation:
        # get all fields except loss_mask
        new_frame = {k: v for k, v in frame.items() if k != "loss_mask"}
        new_sample.append(new_frame)

    sample_text = [new_sample[0]]
    image = new_sample[0]["content"][1]["image"]
    target = new_sample[1]["content"]

    text_input = processor.apply_chat_template(
        sample_text, tokenize=False, add_generation_prompt=True
    )

    input_tokens = processor(
        text=text_input,
        images=[image],
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
