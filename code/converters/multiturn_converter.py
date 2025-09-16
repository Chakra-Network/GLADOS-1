from .base_pango_converter import BasePangoConverter
from ..consts import MIN_PIXELS, MAX_PIXELS


class MultiturnConverter(BasePangoConverter):
    def __init__(
        self,
        dataset_path: str,
        prompt: str,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ):
        super().__init__(dataset_path, prompt, min_pixels, max_pixels)

    def generate_conversation(self, *args, **kwargs):
        raise NotImplementedError(
            "MultiturnConverter does not support generate_conversation"
        )

    def generate_indices(self, n: int, pct_train: float):
        raise NotImplementedError(
            "MultiturnConverter does not support generate_indices"
        )
