from typing import Callable

import torch
import wandb
from ..consts import WANDB_API_KEY

from ..converters.state_transition_converter import StateTransitionConverter
from ..converters.simple_grounding_converter import SimpleGroundingConverter
from ..utils import parse_action
from ast import literal_eval as make_tuple


def wandb_log(data: dict, step: int):
    if WANDB_API_KEY:
        wandb.log(data, step=step)


def generate_cross_validation_loss(
    processor,
    model,
    validation_indices: list[int],
    converter: StateTransitionConverter | SimpleGroundingConverter,
    index_to_training_message: Callable,
):
    total_loss = 0

    indices_tested = 0
    model.eval()
    with torch.no_grad():
        for index in validation_indices:
            try:
                input_ids, attention_mask, labels = index_to_training_message(
                    index, converter, processor
                )
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                print(f"Validation loss: {loss.item()}")
                total_loss += loss.item()
                indices_tested += 1
            except Exception as e:
                print(f"Error testing index {index}: {e}")
                continue
    return total_loss / indices_tested


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Find max length in batch
    max_length = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]

        # Pad sequences to max_length
        pad_length = max_length - input_ids.size(0)
        if pad_length > 0:
            # Create padding tensors on the same device as the original tensors
            pad_input_ids = torch.full(
                (pad_length,), 0, device=input_ids.device, dtype=input_ids.dtype
            )
            pad_attention_mask = torch.zeros(
                pad_length, device=attention_mask.device, dtype=attention_mask.dtype
            )
            pad_labels = torch.full(
                (pad_length,), -100, device=labels.device, dtype=labels.dtype
            )

            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
    }


def try_get_coordinates_for_target_and_predicted_response(
    target_mask, target_tokens, processor, output_logits_for_sample, device=None
) -> dict[str, tuple[float, float]]:
    """
    Try to get the coordinates for the target and predicted response.
    If the target response does not contain "Action:", raise an AssertionError.
    Otherwise, tries to parse the action from the target response.
    If the predicted response does not contain "Action:", raises a Value Error.

    returns dictionary of format:
    {
        "target_coordinates": (float, float),
        "predicted_coordinates": (float, float),
    }
    """
    target_response = processor.batch_decode([target_tokens], skip_special_tokens=True)[
        0
    ]

    print(f"Target: {target_response}")

    # Get model's actual predictions
    # Take argmax of logits to get predicted token IDs
    predicted_token_ids = torch.argmax(output_logits_for_sample, dim=-1)

    # Fix alignment: model at position i predicts token at position i+1
    # So we need to shift the predictions to align with targets
    target_positions = torch.where(target_mask)[0]

    # For each target position, get the prediction from the previous position
    aligned_predictions = []
    for pos in target_positions:
        if pos > 0:  # Can't predict first token from position -1
            pred_token_id = predicted_token_ids[pos - 1]
            aligned_predictions.append(pred_token_id)
        else:
            # For the first position, we can't get a prediction, so skip or use a placeholder
            aligned_predictions.append(processor.tokenizer.unk_token_id)

    # Convert to tensor for batch decoding
    if aligned_predictions:
        predicted_tokens = torch.tensor(aligned_predictions, device=device)
        predicted_response = processor.batch_decode(
            [predicted_tokens], skip_special_tokens=True
        )[0]
    else:
        predicted_response = "[No predictions to decode]"

    print(f"Predicted: {predicted_response}")
    # Compare parsed actions if both contain "Action:"
    assert (
        "Action:" in target_response
    ), "Target response does not contain 'Action:' {target_response}. Malformatted action."

    target_parsed_action = parse_action(target_response.split("Action:")[-1].lstrip())

    if not "Action:" in predicted_response:
        raise ValueError(
            "Predicted response does not contain 'Action:' {predicted_response}. Malformatted action."
        )

    predicted_parsed_action = parse_action(
        predicted_response.split("Action:")[-1].lstrip()
    )

    def _is_valid_numeric_tuple(tuple):
        return len(tuple) == 2 and all(
            isinstance(x, float) or isinstance(x, int) for x in tuple
        )

    # Calculate coordinate difference if both have coordinates
    if "start_box" in target_parsed_action.get(
        "args", {}
    ) and "start_box" in predicted_parsed_action.get("args", {}):
        target_coords = make_tuple(target_parsed_action["args"]["start_box"])
        pred_coords = make_tuple(predicted_parsed_action["args"]["start_box"])

        if not _is_valid_numeric_tuple(target_coords) or not _is_valid_numeric_tuple(
            pred_coords
        ):
            raise ValueError(
                "Target or predicted coordinates are not valid float tuple"
            )

        return {
            "target_coordinates": target_coords,
            "predicted_coordinates": pred_coords,
        }
