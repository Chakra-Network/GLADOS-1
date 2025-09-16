import argparse
import re
from code.consts import (
    GROUNDING_PROMPT,
    MAX_STANDARDIZED_X_COORDINATE,
    MAX_STANDARDIZED_Y_COORDINATE,
)
from code.converters.simple_grounding_converter import SimpleGroundingConverter


class TestGroundingConverter:

    def __init__(self, is_sample: bool = True):
        self.dataset_path = (
            "chakra-labs/pango-sample" if is_sample else "chakra-labs/pango"
        )
        self.converter = SimpleGroundingConverter(
            self.dataset_path, prompt=GROUNDING_PROMPT
        )
        self.train_indices, self.test_indices = self.converter.generate_indices(
            n=10, pct_train=0.8
        )

    def test_generate_indices(self):
        assert len(self.train_indices) > 0, "Should generate at least one sample"
        assert len(self.test_indices) > 0, "Should generate at least one sample"
        assert (
            len(self.train_indices) + len(self.test_indices) == 10
        ), "Should generate 10 samples"
        assert len(self.train_indices) == 8, "Should generate 8 train samples"
        assert len(self.test_indices) == 2, "Should generate 2 test samples"

    def test_all_valid_box_coordinates(self):
        for index in self.train_indices + self.test_indices:
            convo = self.converter.generate_conversation(self.converter.actions[index])
            action_msg = convo[-1]["content"]
            match = re.search(
                r"start_box='<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>'",
                action_msg,
            )
            print(action_msg)
            assert (
                match is not None
            ), "Action message should contain valid box coordinates"
            x, y = match.groups()
            assert (
                float(x) >= 0 and float(x) <= MAX_STANDARDIZED_X_COORDINATE
            ), "X coordinate should be between 0 and 1"
            assert (
                float(y) >= 0 and float(y) <= MAX_STANDARDIZED_Y_COORDINATE
            ), "Y coordinate should be between 0 and 1"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sample", action="store_true", default=False)
    args = args.parse_args()
    print("#" * 50)
    print(f"Testing grounding with {'sample' if args.sample else 'all'} dataset...")
    print("#" * 50)
    test_instance = TestGroundingConverter(args.sample)
    print("\nTesting all valid box coordinates...\n")
    test_instance.test_all_valid_box_coordinates()
    print("\nTesting generate indices...\n")
    test_instance.test_generate_indices()
