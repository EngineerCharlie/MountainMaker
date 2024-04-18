import cv2
import numpy as np


def reduce_colors(image, num_colors=3):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Convert to float32
    pixels = np.float32(pixels)

    # Define criteria (number of iterations and epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Apply k-means clustering
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


def flood_fill(image, seed_point, target_color):
    h, w = image.shape[:2]
    seed_color = np.array(image[seed_point[0], seed_point[1]])
    visited = np.zeros((h, w), dtype=bool)
    stack = [seed_point]

    while stack:
        x, y = stack.pop()

        if x < 0 or x >= w or y < 0 or y >= h or visited[y, x]:
            continue
        if np.array_equal(image[y, x], target_color) or not np.array_equal(
            image[y, x], seed_color
        ):
            continue

        image[y, x] = target_color
        visited[y, x] = True

        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))
    return image


def resize_and_center_crop(image_path, target_width=800, target_height=600):
    # Read the image
    image = cv2.imread(image_path)
    return resize_and_center_crop_image(image, target_width, target_height)


def resize_and_center_crop_image(image, target_width=800, target_height=600):

    # Get the dimensions of the original image
    original_height, original_width = image.shape[:2]
    # TODO: Should it upscale or should it discard upscales?

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height  # 16:9

    # Calculate the aspect ratio of the target size
    target_aspect_ratio = target_width / target_height  # 4:3

    # Determine whether to resize based on width or height
    if aspect_ratio >= target_aspect_ratio:
        # old image is height limited so resize based on height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        # Resize based on width
    else:
        # Old image is width limited so resize based on width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate the cropping parameters
    crop_x = max(0, (new_width - target_width) // 2)
    crop_y = max(0, (new_height - target_height) // 2)

    # Perform the centered crop
    cropped_image = resized_image[
        crop_y : crop_y + target_height, crop_x : crop_x + target_width
    ]

    return cropped_image


def load_and_save_scaled_image(inputPath, outputPath):
    resizedImage = resize_and_center_crop(inputPath)
    # Save the resized image
    cv2.imwrite(outputPath, resizedImage)
