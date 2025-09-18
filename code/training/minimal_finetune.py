import math
import os
from datetime import datetime

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from ..consts import (
    GROUNDING_PROMPT,
    MAX_PIXELS,
    MIN_PIXELS,
    STATE_TRANSITION_PROMPT,
    WANDB_API_KEY,
)
from ..converters.simple_grounding_converter import SimpleGroundingConverter
from ..converters.state_transition_converter import StateTransitionConverter
from ..datasets.compliance import ComplianceSampleDataset
from ..datasets.grounding import GroundingSampleDataset
from ..datasets.grounding import convert_grounding_index_to_training_message
from ..datasets.compliance import convert_compliance_index_to_training_message
from ..datasets.state_transition import StateTransitionSampleDataset
from ..datasets.state_transition import (
    convert_state_transition_index_to_training_message,
)
from ..exceptions import ActionConversionError
from .checkpoint import check_cuda, load_checkpoint, save_checkpoint, upload_checkpoint
from .helpers import (
    generate_cross_validation_loss,
    collate_fn,
    try_get_coordinates_for_target_and_predicted_response,
    wandb_log,
)

input_model = "chakra-labs/pango-7b-sft-checkpoints-state-transition"
config_url = "https://huggingface.co/ByteDance-Seed/UI-TARS-7B-SFT/raw/main/config.json"
checkpoint_model_name = "chakra-labs/pango-7b-sft-compliance"
model_name = checkpoint_model_name[checkpoint_model_name.rfind("/") + 1 :]
output_dir = "checkpoints/epoch_{epoch}_" + model_name


# hyperparameters
training_sample_size = 1000
validation_sample_size = 100
learning_rate = 1e-5
batch_size = 4
# Effective batch size = batch_size * gradient_accumulation_steps
gradient_accumulation_steps = 2
# Learning rate annealing feature flag (default False)
# When enabled, uses AdamW + scheduler; when disabled, uses SGD for memory efficiency
lr_annealing_enabled = False
# See each training sample once per epoch (training_sample_size/ batch_size * gradient_accumulation_steps)
steps_per_epoch = int(training_sample_size / (batch_size * gradient_accumulation_steps))
# Total number of steps = num_epochs * steps_per_epoch = training_sample_size. Should be equal to training_sample_size empirically
# must be an integer
num_epochs = 10
MAX_DISTANCE_POSSIBLE = math.sqrt(2_000_000)


def finetune(
    processor,
    model: PreTrainedModel,
    converter: SimpleGroundingConverter | StateTransitionConverter | None,
    training_indices: list[int],
    validation_indices: list[int],
    dataset_type: str = "grounding",
    commit_hash: str = None,
):
    # Get wandb API key from environment variable
    if WANDB_API_KEY:

        wandb.login(key=WANDB_API_KEY)

        # Initialize wandb
        run_name = f"{dataset_type}_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if commit_hash:
            run_name += f"_{commit_hash}"

        wandb.init(
            project="pango-7b-sft",
            name=run_name,
            config={
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "steps_per_epoch": steps_per_epoch,
                "lr_annealing_enabled": lr_annealing_enabled,
                "model": "chakra-labs/pango-7b-sft-checkpoints",
                "commit_hash": commit_hash,
            },
        )

    # Select appropriate conversion function based on dataset type
    if dataset_type == "grounding":
        convert_fn = convert_grounding_index_to_training_message
    elif dataset_type == "state_transition":
        convert_fn = convert_state_transition_index_to_training_message
    elif dataset_type == "compliance":
        convert_fn = convert_compliance_index_to_training_message
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"{dataset_type.title()} finetuning with {len(training_indices)} samples")
    cross_validation_loss = generate_cross_validation_loss(
        processor,
        model,
        validation_indices,
        converter,
        convert_fn,
    )

    print(f"Baseline cross validation loss: {cross_validation_loss:.4f}")
    wandb_log(
        {"train/cross_validation_loss": cross_validation_loss},
        step=0,
    )

    # Create dataset and dataloader based on dataset type
    if dataset_type == "grounding":
        train_dataset = GroundingSampleDataset(training_indices, converter, processor)
    elif dataset_type == "state_transition":
        train_dataset = StateTransitionSampleDataset(
            training_indices, converter, processor
        )
    elif dataset_type == "compliance":
        train_dataset = ComplianceSampleDataset(len(training_indices), processor)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Set model to training mode
    model.train()

    # Define optimizer based on lr_annealing_enabled flag
    if lr_annealing_enabled:
        # AdamW works better with learning rate scheduling
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        # SGD with momentum for memory efficiency when using fixed learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Optionally create learning rate scheduler based on feature flag
    scheduler = None
    if lr_annealing_enabled:
        # Cosine annealing scheduler - reduces LR over time
        total_steps = num_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=learning_rate * 0.1
        )

    # Training loop
    print("Starting training steps...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        step_count = 0
        compliant_action_format_in_epoch_count = 0

        # Create iterator that cycles through the dataloader
        dataloader_iter = iter(train_dataloader)

        for i in range(steps_per_epoch):
            # Get next batch, cycling through dataset if needed
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
            except OSError as e:
                batch = None
                while batch is None:
                    print(f"\n\nOSERROR: {e}\n\n")
                    batch = next(dataloader_iter)
            except ActionConversionError as e:
                batch = None
                while batch is None:
                    print(f"\n\nActionConversionError: {e}\n\n")
                    batch = next(dataloader_iter)
            except Exception as e:
                print(f"\n\nException: {e}\n\n")
                continue

            # Move batch to GPU
            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

                total_euclidean_distance_for_batch = 0
                compliant_action_format_in_batch_count = 0

                # Compare model predictions with targets
                for j in range(batch["input_ids"].shape[0]):
                    euclidean_distance = MAX_DISTANCE_POSSIBLE
                    # Get target labels (what the model should predict)
                    target_mask = batch["labels"][j] != -100  # Find non-ignored tokens
                    assert target_mask.any(), "Target mask is all False"
                    target_tokens = batch["labels"][j][target_mask]
                    output_logits_for_sample = outputs.logits[j]
                    device = batch["input_ids"].device
                    try:
                        coordinates_dict = (
                            try_get_coordinates_for_target_and_predicted_response(
                                target_mask,
                                target_tokens,
                                processor,
                                output_logits_for_sample,
                                device,
                            )
                        )
                        target_coordinates = coordinates_dict["target_coordinates"]
                        predicted_coordinates = coordinates_dict[
                            "predicted_coordinates"
                        ]

                        coord_diff = math.sqrt(
                            (target_coordinates[0] - predicted_coordinates[0]) ** 2
                            + (target_coordinates[1] - predicted_coordinates[1]) ** 2
                        )
                        if coord_diff < MAX_DISTANCE_POSSIBLE:
                            euclidean_distance = coord_diff

                        compliant_action_format_in_epoch_count += 1
                        compliant_action_format_in_batch_count += 1
                    except AssertionError as e:
                        print("Critical error: ", e)
                        raise e
                    except (ValueError, Exception) as e:
                        print("Malformatted action: ", e)

                    total_euclidean_distance_for_batch += euclidean_distance

                compliance_rate_in_batch = float(
                    compliant_action_format_in_batch_count
                ) / float(batch_size)

                print(
                    f"Total Euclidean distance for batch: {total_euclidean_distance_for_batch}"
                )
                print(
                    f"Average Euclidean distance for batch: {total_euclidean_distance_for_batch / batch_size}"
                )
                wandb_log(
                    {
                        "train/total_euclidean_distance_for_batch": total_euclidean_distance_for_batch,
                        "train/average_euclidean_distance_for_batch": total_euclidean_distance_for_batch
                        / batch_size,
                        "train/compliance_rate_in_batch": compliance_rate_in_batch,
                    },
                    step=epoch * steps_per_epoch + i + 1,
                )

            loss.backward()

            # Gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()  # Update learning rate only if annealing is enabled
                optimizer.zero_grad()
                step_count += 1

                # Clear cache every few steps to prevent memory fragmentation
                if step_count % 5 == 0:
                    torch.cuda.empty_cache()

            total_loss += outputs.loss.item()

            step = epoch * steps_per_epoch + i + 1
            # Log learning rate based on whether scheduler is enabled
            current_lr = (
                scheduler.get_last_lr()[0] if scheduler is not None else learning_rate
            )

            wandb_log(
                {
                    "train/loss": loss.item(),
                    "train/epoch": epoch + 1,
                    "train/step": i + 1,
                    "train/learning_rate": current_lr,
                },
                step=step,
            )

        avg_loss = total_loss / steps_per_epoch
        cross_validation_loss = generate_cross_validation_loss(
            processor,
            model,
            validation_indices,
            converter,
            convert_fn,
        )
        compliance_rate_in_epoch = float(
            compliant_action_format_in_epoch_count
        ) / float(steps_per_epoch * batch_size)

        print(
            f"""
            Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, 
            Cross Validation Loss: {cross_validation_loss:.4f}, 
            Action Format Compliance Rate: {compliance_rate_in_epoch:.4f}
            """
        )

        save_checkpoint(processor, model, config_url, output_dir.format(epoch=epoch))

        wandb_log(
            {
                "train/avg_loss_epoch": avg_loss,
                "train/epoch": epoch + 1,
                "train/cross_validation_loss": cross_validation_loss,
                "train/compliance_rate_epoch": compliance_rate_in_epoch,
            },
            step=(epoch + 1) * steps_per_epoch,
        )

    print("Minimal finetuning completed!")

    print("Logging response finetuned")

    if WANDB_API_KEY:
        wandb.finish()

    return model


def train(
    dataset_type: str = "grounding",
    commit_hash: str = None,
    is_pango_sample: bool = True,
):
    # Enable memory optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    check_cuda()
    processor, model, _ = load_checkpoint(input_model, config_url=config_url)
    total_sample_size = training_sample_size + validation_sample_size
    dataset_path = (
        "chakra-labs/pango-sample" if is_pango_sample else "chakra-labs/pango"
    )

    # Create converter based on dataset type
    if dataset_type == "grounding":
        converter = SimpleGroundingConverter(
            dataset_path=dataset_path,
            prompt=GROUNDING_PROMPT,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
    elif dataset_type == "state_transition":
        converter = StateTransitionConverter(
            dataset_path=dataset_path,
            prompt=STATE_TRANSITION_PROMPT,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
    elif dataset_type == "compliance":
        training_indices = list(range(training_sample_size))
        validation_indices = list(range(training_sample_size, total_sample_size))
        converter = None
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    print(f"Training indices: {len(training_indices)}")
    print(f"Validation indices: {len(validation_indices)}")
    finetuned_model = finetune(
        processor,
        model,
        converter,
        training_indices,
        validation_indices,
        dataset_type,
        commit_hash,
    )
    upload_checkpoint(
        processor,
        finetuned_model,
        output_dir.format(epoch=num_epochs - 1),
        checkpoint_model_name,
    )
