from code.converters.state_transition_converter import StateTransitionConverter
from code.consts import (
    STATE_TRANSITION_PROMPT,
    MAX_STANDARDIZED_X_COORDINATE,
    MAX_STANDARDIZED_Y_COORDINATE,
)
from PIL import Image
from qwen_vl_utils import fetch_image
import matplotlib.pyplot as plt


def visualize_action(converter: StateTransitionConverter, action: dict):
    raw_image = Image.open(converter._get_screenshot_path(action))
    transformed_image = fetch_image(
        {
            "image": raw_image,
            "min_pixels": converter.min_pixels,
            "max_pixels": converter.max_pixels,
        }
    )
    print("raw_image.size", raw_image.size)
    print("transformed_image.size", transformed_image.size)

    click_x = action["x"]
    click_y = action["y"]
    print(f"Original click coordinates: ({click_x}, {click_y})")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(raw_image)
    ax1.set_title(f"Raw Image ({raw_image.size[0]}x{raw_image.size[1]})")
    ax1.axis("off")
    ax1.plot(
        click_x,
        click_y,
        "ro",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
    )
    ax1.text(
        click_x + 10,
        click_y + 10,
        f"({click_x}, {click_y})",
        color="red",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax2.imshow(transformed_image)
    ax2.set_title(
        f"Transformed Image ({transformed_image.size[0]}x{transformed_image.size[1]})"
    )
    ax2.axis("off")

    # first scale the coordinates to 1000x1000
    standardized_x, standardized_y = converter._scale_coordinates(
        click_x, click_y, raw_image.size, transformed_image.size
    )
    # then scale them back to the transformed image dimensions
    transformed_x, transformed_y = (
        standardized_x / MAX_STANDARDIZED_X_COORDINATE * transformed_image.size[0],
        standardized_y / MAX_STANDARDIZED_Y_COORDINATE * transformed_image.size[1],
    )
    print(f"Transformed click coordinates: ({transformed_x:.1f}, {transformed_y:.1f})")

    ax2.plot(
        transformed_x,
        transformed_y,
        "ro",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
    )
    ax2.text(
        transformed_x + 10,
        transformed_y + 10,
        f"({transformed_x:.1f}, {transformed_y:.1f})",
        color="red",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.suptitle("Image Transformation Comparison with Click Coordinates", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    print("Getting actions....")
    converter = StateTransitionConverter(
        dataset_path="chakra-labs/pango-sample",
        prompt=STATE_TRANSITION_PROMPT,
    )
    actions = converter.actions
    for action in actions:
        visualize_action(converter, action)


if __name__ == "__main__":
    main()
