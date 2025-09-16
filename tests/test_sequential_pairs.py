from code.converters.state_transition_converter import StateTransitionConverter
from code.consts import (
    STATE_TRANSITION_PROMPT,
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def create_difference_heatmap(img1, img2):
    """Create a heatmap showing differences between two PIL images."""
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # ensure images are the same size
    if img1_array.shape != img2_array.shape:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        img2_array = np.array(img2)

    # calculate absolute difference
    diff = cv2.absdiff(img1_array, img2_array)

    # convert to grayscale for intensity mapping
    if len(diff.shape) == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    else:
        diff_gray = diff

    # apply Gaussian blur to smooth the heatmap
    diff_smooth = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    # apply colormap for heatmap visualization
    heatmap = cv2.applyColorMap(diff_smooth, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap_rgb, diff_smooth


def display_pair(converter: StateTransitionConverter, pair: tuple[dict, dict]):
    current_action, next_action = pair
    current_image = Image.open(converter._get_screenshot_path(current_action))
    next_image = Image.open(converter._get_screenshot_path(next_action))

    print("current_image.size", current_image.size)
    print("next_image.size", next_image.size)

    click_x = current_action["x"]
    click_y = current_action["y"]
    print(f"Click coordinates: ({click_x}, {click_y})")

    heatmap, diff_intensity = create_difference_heatmap(current_image, next_image)

    total_pixels = diff_intensity.size
    changed_pixels = np.count_nonzero(diff_intensity > 10)
    change_percentage = (changed_pixels / total_pixels) * 100
    max_change = np.max(diff_intensity)

    print(
        f"Change statistics: {change_percentage:.1f}% pixels changed, max intensity: {max_change}"
    )

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    ax1.imshow(current_image)
    ax1.set_title(f"Current State ({current_image.size[0]}x{current_image.size[1]})")
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

    ax2.imshow(next_image)
    ax2.set_title(f"Next State ({next_image.size[0]}x{next_image.size[1]})")
    ax2.axis("off")

    ax3.imshow(heatmap)
    ax3.set_title(f"Difference Heatmap\n{change_percentage:.1f}% changed")
    ax3.axis("off")

    ax3.plot(
        click_x,
        click_y,
        "wo",
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=1,
    )

    plt.suptitle("State Transition with Change Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    converter = StateTransitionConverter(
        dataset_path="chakra-labs/pango-sample",
        prompt=STATE_TRANSITION_PROMPT,
    )
    pairs = converter.get_sequentially_safe_pairs()
    for pair in pairs:
        display_pair(converter, pair)


if __name__ == "__main__":
    main()
