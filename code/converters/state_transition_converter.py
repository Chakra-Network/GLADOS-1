import random

from PIL import Image
from qwen_vl_utils import fetch_image

from ..consts import MAX_PIXELS, MIN_PIXELS
from ..converters.base_pango_converter import BasePangoConverter
from ..exceptions import ActionConversionError


class StateTransitionConverter(BasePangoConverter):
    def __init__(
        self,
        dataset_path: str,
        prompt: str,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ):
        # only include click events
        actions_to_ignore = [
            "mouseover_start",
            "mouseover_end",
            "drag_start",
            "drag_end",
            "key_press",
            "input",
            "scroll",
        ]
        super().__init__(
            dataset_path,
            actions_to_ignore=actions_to_ignore,
            download_all_images=True,  # download all images so the next screenshot is always available
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.prompt = prompt
        self.pairs = self.get_sequentially_safe_pairs()

    def _get_action_content(
        self,
        action: dict,
        original_dimensions: tuple[int, int],
        scaled_dimensions: tuple[int, int],
    ):
        action_type = action["type"]
        if action_type == "click":
            return self._handle_click(action, original_dimensions, scaled_dimensions)
        raise ActionConversionError(f"Unknown action type {action_type}")

    def _handle_click(
        self,
        action: dict,
        original_dimensions: tuple[int, int],
        scaled_dimensions: tuple[int, int],
    ):
        x, y = self._scale_coordinates(
            action["x"], action["y"], original_dimensions, scaled_dimensions
        )
        button = action["button"]
        click_count = action["click_count"]

        if button == "button_left" and click_count == 1:
            action_content = f"click(start_box='<|box_start|>({x}, {y})<|box_end|>')"
        elif button == "button_left" and click_count >= 2:
            action_content = (
                f"left_double(start_box='<|box_start|>({x}, {y})<|box_end|>')"
            )
        elif button == "button_right":
            action_content = (
                f"right_single(start_box='<|box_start|>({x}, {y})<|box_end|>')"
            )
        else:
            raise ActionConversionError(f"Unknown click action: {action}")

        return action_content

    def _handle_scroll(
        self,
        action: dict,
        original_dimensions: tuple[int, int],
        scaled_dimensions: tuple[int, int],
    ):
        x, y = self._scale_coordinates(
            action["x"], action["y"], original_dimensions, scaled_dimensions
        )
        direction = self._get_scroll_direction(action)

        action_content = f"scroll(start_box='<|box_start|>({x},{y})<|box_end|>' direction='{direction}')"
        return action_content

    def generate_conversation(self, current_action: dict, next_action: dict):
        current_screenshot_path = self._get_screenshot_path(current_action)
        next_screenshot_path = self._get_screenshot_path(next_action)
        try:
            current_img = Image.open(current_screenshot_path)
            next_img = Image.open(next_screenshot_path)
            original_dimensions = current_img.size
            current_img = fetch_image(
                {
                    "image": current_img,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
            scaled_dimensions = current_img.size
            action_content = self._get_action_content(
                current_action, original_dimensions, scaled_dimensions
            )
            next_img = fetch_image(
                {
                    "image": next_img,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is the current state of the GUI.",
                        },
                        {"type": "image", "image": current_img},
                    ],
                    "loss_mask": 0,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is the next state of the GUI.",
                        },
                        {"type": "image", "image": next_img},
                    ],
                    "loss_mask": 0,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                    ],
                    "loss_mask": 0,
                },
                {
                    "role": "assistant",
                    "content": f"Action: {action_content}",
                    "loss_mask": 1,
                },
            ]
        except Exception as e:
            self._handle_error(
                True,
                f"Error generating conversation: {e}.\n\n{current_action}\n\n{next_action}\n\n",
            )

    def generate_indices(self, n: int, pct_train: float):
        if n > len(self.pairs):
            raise ValueError(
                f"n must be less than or equal to the number of pairs: {len(self.pairs)}"
            )
        random.shuffle(self.pairs)
        indices = list(range(len(self.pairs)))
        random.shuffle(indices)
        num_train = round(n * pct_train)
        num_test = n - num_train
        train_indices = indices[:num_train]
        test_indices = indices[num_train : num_train + num_test]
        return train_indices, test_indices
