import re
import argparse

from code.consts import (
    STATE_TRANSITION_PROMPT,
    MAX_STANDARDIZED_X_COORDINATE,
    MAX_STANDARDIZED_Y_COORDINATE,
)
from code.converters.state_transition_converter import StateTransitionConverter


class TestStateTransitionConverter:

    def __init__(self, is_sample: bool = True):
        self.dataset_path = (
            "chakra-labs/pango-sample" if is_sample else "chakra-labs/pango"
        )
        self.converter = StateTransitionConverter(
            self.dataset_path, prompt=STATE_TRANSITION_PROMPT
        )
        self.n = 2000
        self.train_indices, self.test_indices = self.converter.generate_indices(
            n=self.n, pct_train=0.8
        )

    def test_state_transition_indices_generation(self):
        assert len(self.train_indices) > 0, "Should generate at least one sample"
        assert len(self.test_indices) > 0, "Should generate at least one sample"
        assert (
            len(self.train_indices) + len(self.test_indices) == self.n
        ), f"Should generate {self.n} indices"
        assert len(self.train_indices) == int(
            self.n * 0.8
        ), f"Should generate {self.n * 0.8} train indices"
        assert len(self.test_indices) == int(
            self.n * 0.2
        ), f"Should generate {self.n * 0.2} test indices"

    def test_all_valid_box_coordinates(self):
        for index in self.train_indices + self.test_indices:
            current_action, next_action = self.converter.pairs[index]
            action_msg = self.converter.generate_conversation(
                current_action, next_action
            )[-1]["content"]
            match = re.search(
                r"start_box='<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>'",
                action_msg,
            )
            assert (
                match is not None
            ), "Action message should contain valid box coordinates"
            x, y = match.groups()
            assert (
                float(x) >= 0 and float(x) <= MAX_STANDARDIZED_X_COORDINATE
            ), f"X coordinate should be between 0 and {MAX_STANDARDIZED_X_COORDINATE}"
            assert (
                float(y) >= 0 and float(y) <= MAX_STANDARDIZED_Y_COORDINATE
            ), f"Y coordinate should be between 0 and {MAX_STANDARDIZED_Y_COORDINATE}"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sample", action="store_true", default=False)
    args = args.parse_args()
    print("#" * 50)
    print(
        f"Testing state transition with {'sample' if args.sample else 'all'} dataset..."
    )
    print("#" * 50)
    test_instance = TestStateTransitionConverter(args.sample)
    print("\nTesting all valid box coordinates...")
    test_instance.test_all_valid_box_coordinates()
    print("All tests passed!\n")
    print("\nTesting state transition indices generation...")
    test_instance.test_state_transition_indices_generation()
    print("All tests passed!\n")
