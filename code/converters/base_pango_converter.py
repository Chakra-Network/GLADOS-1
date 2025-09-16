import json
import os
import re
from abc import ABC, abstractmethod

import requests
from datasets import load_dataset

from ..consts import (
    MAX_PIXELS,
    MAX_STANDARDIZED_X_COORDINATE,
    MAX_STANDARDIZED_Y_COORDINATE,
    MIN_PIXELS,
    STORAGE_DIR,
    UUID_REGEX,
)
from ..utils import download_images
from PIL import ImageFile

# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BasePangoConverter(ABC):

    MUST_IGNORE_ACTIONS = ["mouseover_start", "mouseover_end"]

    def __init__(
        self,
        dataset_path: str,
        actions_to_ignore: list[str],
        fail_gracefully: bool = True,
        redownload_actions: bool = False,
        download_all_images: bool = False,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ):
        self.dataset_path = dataset_path
        self.actions_to_ignore = actions_to_ignore
        self.fail_gracefully = fail_gracefully
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.download_all_images = download_all_images

        self.storage_dir = STORAGE_DIR
        self.raw_actions_path = f"{self.storage_dir}/{dataset_path.split('/')[-1]}.json"
        self.df = None
        self.malformatted_actions = []

        os.makedirs(self.storage_dir, exist_ok=True)
        if os.path.exists(self.raw_actions_path) and not redownload_actions:
            self.raw_actions = json.load(open(self.raw_actions_path, "r"))
        else:
            self.raw_actions = self._fetch_actions()
            json.dump(self.raw_actions, open(self.raw_actions_path, "w"))
        self.actions = self._clean_actions()

        self._download_images()

    def _handle_error(self, is_error: bool, message: str):
        if not self.fail_gracefully and is_error:
            raise ValueError(message)
        if is_error:
            print("[GRACEFULLY_FAILED]", message)
        return is_error

    def _handle_malformatted_action(
        self,
        is_error: bool,
        message: str,
        action: dict,
    ):
        if self._handle_error(is_error, message):
            self.malformatted_actions.append(action)
        return is_error

    def _fetch_actions(self):
        dataset = load_dataset(self.dataset_path)
        self.df = dataset["train"].to_pandas()
        self.df = self.df[
            ["id", "synthetically_generated_instruction", "input_metadata"]
        ]
        raw_actions = []
        session_count = 0
        for _, url in self.df[["id", "input_metadata"]].values:
            response = requests.get(url)
            raw_actions.extend(response.json()["actions"])
            session_count += 1
            print(
                f"{session_count} / {self.df.shape[0]} sessions remaining. {len(raw_actions)} raw actions loaded."
            )
        return self._enhace_raw_actions(raw_actions)

    def _filter_actions(self):
        return [
            action
            for action in self.raw_actions
            if action["type"] not in self.actions_to_ignore
        ]

    def _enhace_raw_actions(self, actions: list[dict]):
        enhanced_actions = []
        for action in actions:
            if action["type"] in self.MUST_IGNORE_ACTIONS:
                continue
            action["session_id"] = self._get_session_id_from_action(action)
            action["index"] = len(enhanced_actions)
            enhanced_actions.append(action)
        return enhanced_actions

    def _convert_drag_actions(self, actions: list[dict]):
        new_actions = []
        for idx, action in enumerate(actions):
            if action["type"] == "drag_start":
                if idx + 1 < len(actions) and actions[idx + 1]["type"] == "drag_end":
                    is_current_malformatted = (
                        action.get("x") is None or action.get("y") is None
                    )
                    is_next_malformatted = (
                        actions[idx + 1].get("x") is None
                        or actions[idx + 1].get("y") is None
                    )
                    if self._handle_malformatted_action(
                        is_current_malformatted,
                        "Malformatted drag start action",
                        action,
                    ) or self._handle_malformatted_action(
                        is_next_malformatted,
                        "Malformatted drag end action",
                        actions[idx + 1],
                    ):
                        continue
                    new_actions.append(
                        {
                            "type": "drag",
                            "start_x": action["x"],
                            "end_x": actions[idx + 1]["x"],
                            "start_y": action["y"],
                            "end_y": actions[idx + 1]["y"],
                            "screenshot_url": action["screenshot_url"],
                            "session_id": action["session_id"],
                            "index": action["index"] + 1,
                            "relative_timestamp_ms": action["relative_timestamp_ms"],
                        }
                    )
                else:
                    if self._handle_malformatted_action(
                        True, "Drag start and drag end must be paired", action
                    ):
                        continue
            elif action["type"] == "drag_end":
                continue
            else:
                new_actions.append(action)
        return new_actions

    def _clean_actions(self):
        cleaned_actions = self._convert_drag_actions(self._filter_actions())
        num_cleaned = len(cleaned_actions)
        print(f"There are {num_cleaned} actions after cleaning.")
        return cleaned_actions

    def _scale_coordinates(
        self,
        x: int,
        y: int,
        original_dimensions: tuple[int, int],
        scaled_dimensions: tuple[int, int],
    ):
        width_factor = scaled_dimensions[0] / original_dimensions[0]
        height_factor = scaled_dimensions[1] / original_dimensions[1]
        scaled_x = x * width_factor
        scaled_y = y * height_factor
        return int(
            scaled_x / scaled_dimensions[0] * MAX_STANDARDIZED_X_COORDINATE
        ), int(scaled_y / scaled_dimensions[1] * MAX_STANDARDIZED_Y_COORDINATE)

    def _get_session_id_from_action(self, action: dict):
        return re.search(UUID_REGEX, action["screenshot_url"]).group(1)

    def _get_screenshot_path(self, action: dict):
        id_ = action["session_id"]
        return f"{self.storage_dir}/{id_}_{action['relative_timestamp_ms']}.png"

    def _download_images(self):
        if self.download_all_images:
            download_images(
                self.storage_dir, self.raw_actions_path, max_concurrent=1000
            )
        else:
            download_path = f"{self.storage_dir}/to_download.json"
            json.dump(self.actions, open(download_path, "w"))
            download_images(self.storage_dir, download_path, max_concurrent=1000)
            os.remove(download_path)

    def get_sequentially_safe_pairs(
        self,
    ) -> list[tuple[dict, dict]]:
        print("Getting sequentially safe pairs")
        safe_pairs = []
        actions = self.raw_actions.copy()
        # ensure actions are sorted by index in ascending order
        actions.sort(key=lambda x: x["index"])
        for i in range(len(actions) - 1):
            try:
                if actions[i]["type"] in self.actions_to_ignore:
                    continue
                if actions[i]["session_id"] != actions[i + 1]["session_id"]:
                    continue
                if actions[i + 1]["index"] != actions[i]["index"] + 1:
                    continue
                safe_pairs.append((actions[i], actions[i + 1]))
            except Exception as e:
                self._handle_error(
                    True,
                    f"Error getting sequentially safe pairs: {e}.\n\n{actions[i]}\n\n{actions[i + 1]}\n\n",
                )
        print(f"Found {len(safe_pairs)} sequentially safe pairs")
        return safe_pairs

    @abstractmethod
    def generate_conversation(self, *args, **kwargs) -> list:
        """
        Generates a conversation between two actions
        Args:
            *args: arguments
            **kwargs: keyword arguments
        Returns:
            conversation: list of conversation
        """
        pass

    @abstractmethod
    def generate_indices(self, n: int, pct_train: float) -> tuple[list, list]:
        """
        Generates indices for train and test sets
        n: number of indices to generate
        pct_train: percentage of indices to put in the train set
        Returns:
            train_indices: list of train indices
            test_indices: list of test indices
        """
        pass
