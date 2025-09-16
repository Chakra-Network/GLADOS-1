import random

from PIL import Image
from qwen_vl_utils import fetch_image

from ..consts import MAX_PIXELS, MIN_PIXELS
from ..exceptions import ActionConversionError
from .base_pango_converter import BasePangoConverter


class SimpleGroundingConverter(BasePangoConverter):
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
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.prompt = prompt
        self.actions_with_downloaded_images = []

    def _get_action_content_and_instruction(
        self,
        action: dict,
        original_dimensions: tuple[int, int],
        scaled_dimensions: tuple[int, int],
    ):
        action_type = action["type"]

        if action_type == "click":
            return self._handle_click(action, original_dimensions, scaled_dimensions)
        raise ActionConversionError(f"Unknown action type {action_type}")

    def __convert_coordinates_to_quadrant(
        self, x: float, y: float, scaled_dimensions: tuple[int, int]
    ):
        x = x / scaled_dimensions[0]
        y = y / scaled_dimensions[1]
        if x < 0.5 and y < 0.5:
            return "top-left"
        elif x < 0.5 and y > 0.5:
            return "bottom-left"
        elif x > 0.5 and y < 0.5:
            return "top-right"
        else:
            return "bottom-right"

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
        quadrant = self.__convert_coordinates_to_quadrant(x, y, scaled_dimensions)

        if button == "button_left" and click_count == 1:
            action_content = f"click(start_box='<|box_start|>({x}, {y})<|box_end|>')"
            instruction = (
                f"Please click on the next logical action in the {quadrant} quadrant"
            )
        elif button == "button_left" and click_count >= 2:
            action_content = (
                f"left_double(start_box='<|box_start|>({x}, {y})<|box_end|>')"
            )
            instruction = f"Please double click on the next logical action in the {quadrant} quadrant"
        elif button == "button_right":
            action_content = (
                f"right_single(start_box='<|box_start|>({x}, {y})<|box_end|>')"
            )
            instruction = f"Please right click on the next logical action in the {quadrant} quadrant"
        else:
            raise ActionConversionError(f"Unknown click action: {action}")

        return action_content, instruction

    def generate_conversation(self, action: dict):
        try:
            screenshot_path = self._get_screenshot_path(action)
            image = Image.open(screenshot_path)
            original_dimensions = image.size
            image = fetch_image(
                {
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
            scaled_dimensions = image.size
            action_content, instruction = self._get_action_content_and_instruction(
                action, original_dimensions, scaled_dimensions
            )
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt.format(instruction=instruction),
                        },
                        {
                            "type": "image",
                            "image": image,
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
        except ActionConversionError as e:
            self._handle_malformatted_action(True, str(e), action)
        except Exception as e:
            self._handle_error(True, f"Error converting action: {e}.\n\n{action}\n\n")

    def generate_indices(self, n: int, pct_train: float):
        if n > len(self.actions):
            raise ValueError(
                f"n must be less than or equal to the number of actions: {len(self.actions)}"
            )
        random.shuffle(self.actions)
        indices = list(range(len(self.actions)))
        random.shuffle(indices)
        num_train = round(n * pct_train)
        num_test = n - num_train
        train_indices = indices[:num_train]
        test_indices = indices[num_train : num_train + num_test]
        return train_indices, test_indices
