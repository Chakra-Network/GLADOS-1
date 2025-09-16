import json
import os
from ..consts import EXAMPLE_DATA_PATH, HF_TOKEN

import requests
import torch
from huggingface_hub import HfApi, login
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
)


def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("No GPU available")
        exit()


def load_config_from_url(url):
    """Load config from a specific Hugging Face URL"""
    print(f"Loading config from {url}")
    response = requests.get(url)
    response.raise_for_status()
    config_dict = response.json()

    # Create config object from the loaded dictionary
    config = Qwen2VLConfig(**config_dict)
    return config


def load_checkpoint(model_name, config_url=None):
    print(f"Loading checkpoint from {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)

    # Load config from URL if provided, otherwise use default
    if config_url:
        config = load_config_from_url(config_url)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        config=config,
    )

    return processor, model, config


def check_inference(processor, model, messages):
    print(f"Checking inference with messages: {messages}")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = processor(text=text, return_tensors="pt").to("cuda")

    output_ids = model.generate(
        input_ids["input_ids"],
        pad_token_id=processor.tokenizer.eos_token_id,
        attention_mask=input_ids["attention_mask"],
        max_new_tokens=150,
    )
    response = processor.batch_decode(
        output_ids[:, input_ids["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    print(f"Response: {response}")
    return response


def load_example_messages():
    with open(f"{EXAMPLE_DATA_PATH}/messages.json", "r") as f:
        return json.load(f)


# HACK: The config that is saved via .save_pretrained does not contain all vision config fields
def __hack_save_config(config_url, output_dir):
    response = requests.get(config_url)
    response.raise_for_status()
    config_dict = response.json()
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)


def __hack_upload_config(repo_id, output_dir):
    api = HfApi()
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="config.json",
        path_or_fileobj=os.path.join(output_dir, "config.json"),
    )


def save_checkpoint(processor, model, config_url, output_dir):
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    __hack_save_config(config_url, output_dir)


def login_huggingface():
    login(token=HF_TOKEN)


def upload_checkpoint(processor, model, output_dir, repo_id):
    def _create_repo():
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, repo_type="model")
        except Exception:
            print("Repo already exists")

    _create_repo()

    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)
    __hack_upload_config(repo_id, output_dir)
    print(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    input_model = "ByteDance-Seed/UI-TARS-7B-SFT"
    config_url = (
        "https://huggingface.co/ByteDance-Seed/UI-TARS-7B-SFT/raw/main/config.json"
    )
    checkpoint_model_name = "chakra-labs/pango-7b-sft-checkpoints"
    output_dir = (
        f"checkpoints/{checkpoint_model_name[checkpoint_model_name.rfind('/') + 1 :]}"
    )

    check_cuda()
    processor, model, config = load_checkpoint(input_model, config_url=config_url)
    example_messages = load_example_messages()
    check_inference(processor, model, example_messages)
    save_checkpoint(processor, model, config_url, output_dir)
    upload_checkpoint(processor, model, output_dir, checkpoint_model_name)
    print(f"Uploaded to https://huggingface.co/{checkpoint_model_name}")
