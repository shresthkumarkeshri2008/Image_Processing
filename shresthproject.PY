"""
Unit 1: Plant Image Processing and Python
Autonomous Drone-Based Crop Health Monitoring - Mini Demonstration

This beginner-friendly program creates a sample leaf image, identifies green
pixels using HSV colour thresholding, calculates the green-pixel percentage,
and displays the original image and the detected region.

Requirements:
    pip install opencv-python numpy matplotlib
Run:
    python unit1_plant_image_processing.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_sample_leaf():
    """Create a simple synthetic leaf image so the demo runs without downloads."""
    image = np.full((400, 500, 3), (225, 235, 245), dtype=np.uint8)  # BGR background
    # Leaf body (ellipse) and central vein
    cv2.ellipse(image, (250, 200), (105, 155), -25, 0, 360, (45, 145, 45), -1)
    cv2.line(image, (195, 330), (305, 70), (220, 235, 220), 5)
    # Side veins
    for y in range(120, 300, 35):
        cv2.line(image, (250, y), (215, y - 18), (190, 220, 190), 3)
        cv2.line(image, (250, y), (285, y - 12), (190, 220, 190), 3)
    return image


def main():
    original = create_sample_leaf()

    # Convert BGR to HSV; HSV is useful for selecting colours.
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # Green colour range. These values work for this generated demonstration.
    lower_green = np.array([30, 35, 25])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    total_pixels = mask.size
    green_pixels = cv2.countNonZero(mask)
    green_percentage = (green_pixels / total_pixels) * 100

    # Keep only the detected green region.
    detected_leaf = cv2.bitwise_and(original, original, mask=mask)

    print("AUTONOMOUS DRONE-BASED CROP HEALTH MONITORING")
    print("UNIT 1: PLANT IMAGE PROCESSING AND PYTHON")
    print("-" * 48)
    print(f"Image size: {original.shape[1]} x {original.shape[0]} pixels")
    print(f"Total pixels: {total_pixels}")
    print(f"Green pixels detected: {green_pixels}")
    print(f"Green pixel percentage: {green_percentage:.2f}%")
    print("Result: Green regions were detected successfully.")
    print("Note: This is a colour-segmentation demo, not a disease diagnosis.")

    # Plot images and save a screenshot-style output figure.
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    detected_rgb = cv2.cvtColor(detected_leaf, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    fig.suptitle("Unit 1 Output: Basic Plant Image Processing", fontsize=14, fontweight="bold")
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original sample leaf")
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Green-pixel mask")
    axes[2].imshow(detected_rgb)
    axes[2].set_title("Detected green region")
    for ax in axes:
        ax.axis("off")
    fig.text(
        0.5, 0.025,
        f"Image: {original.shape[1]} x {original.shape[0]} | "
        f"Green pixels: {green_pixels} | Green percentage: {green_percentage:.2f}%",
        ha="center", fontsize=9
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.90])
    plt.savefig("unit1_code_output_screenshot.png", dpi=160, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
