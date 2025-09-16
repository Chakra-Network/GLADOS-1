import torch
from torch.utils.data import Dataset

from ..converters.state_transition_converter import StateTransitionConverter
from ..exceptions import ActionConversionError


class StateTransitionSampleDataset(Dataset):
    """Custom Dataset for grounding samples with proper batching."""

    def __init__(
        self, indices: list[int], converter: StateTransitionConverter, processor
    ):
        self.indices = indices
        self.converter = converter
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pair_index = self.indices[idx]
        input_ids, attention_mask, labels = (
            convert_state_transition_index_to_training_message(
                pair_index, self.converter, self.processor
            )
        )
        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


def convert_state_transition_index_to_training_message(
    index: int, converter: StateTransitionConverter, processor
):
    """
    Takes a sample (with training masks) and convert it to a tuple with (input_ids, attention_mask, labels)
    """
    # last frame always has a loss mask of 1
    # all other frames have a loss mask of 0

    pair = converter.pairs[index]
    before_action = pair[0]
    after_action = pair[1]
    conversation = converter.generate_conversation(before_action, after_action)

    if not conversation:
        raise ActionConversionError("No conversation generated")

    # first remove loss_mask from the sample
    new_sample = []
    for frame in conversation:
        # get all fields except loss_mask
        new_frame = {k: v for k, v in frame.items() if k != "loss_mask"}
        new_sample.append(new_frame)
    before_image = new_sample[0]["content"][1]["image"]
    after_image = new_sample[1]["content"][1]["image"]
    target = new_sample[-1]["content"]

    text_input = processor.apply_chat_template(
        new_sample[:-1], tokenize=False, add_generation_prompt=True
    )

    input_tokens = processor(
        text=text_input,
        images=[before_image, after_image],
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
