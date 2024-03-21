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
