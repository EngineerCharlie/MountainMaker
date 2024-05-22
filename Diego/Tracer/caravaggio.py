import cv2
import torch
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

# Paths for image input and output
input_images_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/input-images"
depth_images_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/depth-images"
traced_images_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/traced-images"
reduced_depth_images_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/reduced-depth-images"
average_colors_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/average-colors"

# Ensure output directories exist
Path(depth_images_folder).mkdir(parents=True, exist_ok=True)
Path(traced_images_folder).mkdir(parents=True, exist_ok=True)
Path(reduced_depth_images_folder).mkdir(parents=True, exist_ok=True)
Path(average_colors_folder).mkdir(parents=True, exist_ok=True)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("mps")


def convert_image_to_depth_image(img):
    input_tensor = processor(images=img, return_tensors="pt").to("mps")
    with torch.no_grad():
        outputs = model(**input_tensor)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return depth


def get_pixel_coordinates_by_color(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    unique_colors = np.unique(image_rgb.reshape(-1, image_rgb.shape[2]), axis=0)

    if len(unique_colors) != 4:
        raise ValueError("The image does not contain exactly 4 unique colors")

    color_names = ["c1", "c2", "c3", "c4"]
    color_coordinates = {name: [] for name in color_names}
    color_mapping = {tuple(color): name for color, name in zip(unique_colors, color_names)}

    for y in range(image_rgb.shape[0]):
        for x in range(image_rgb.shape[1]):
            pixel_color = tuple(image_rgb[y, x])
            color_name = color_mapping[pixel_color]
            color_coordinates[color_name].append((x, y))

    return color_coordinates


def get_average_color(image, coordinates, filename, color_name):
    cropped_images_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/cropped-images"
    cropped_image_paths = []

    if not coordinates:
        print("No coordinates provided.")
        return 0, 0, 0  # Default to black if no coordinates are provided

    colors = []
    for (x, y) in coordinates:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = image[y, x]
            colors.append(color)

            # Save the cropped portion of the original image
            x1 = max(x - 10, 0)
            x2 = min(x + 10, image.shape[1])
            y1 = max(y - 10, 0)
            y2 = min(y + 10, image.shape[0])
            cropped_image = image[y1:y2, x1:x2]
            cropped_image_path = os.path.join(cropped_images_folder, f"{filename}-{color_name}.png")
            cv2.imwrite(cropped_image_path, cropped_image)
            cropped_image_paths.append(cropped_image_path)
        else:
            print(f"Invalid coordinates: ({x}, {y})")

    if colors:
        average_color = np.mean(colors, axis=0).astype(int)
        print(f"Average color for coordinates: {average_color}")
        return tuple(average_color)
    else:
        print("No valid colors found.")
        return 0, 0, 0  # Default to black if no valid colors are found


def extract_and_replace_colors(img, original_img_colors):
    unique_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    if unique_colors.shape[0] != 4:
        raise ValueError("The input image must contain exactly 4 unique colors.")

    lab_colors = cv2.cvtColor(np.uint8([unique_colors]), cv2.COLOR_RGB2LAB)[0]
    sorted_indices = np.argsort(lab_colors[:, 0])
    sorted_colors = unique_colors[sorted_indices]

    color_mapping = {tuple(sorted_colors[i]): original_img_colors[i] for i in range(4)}

    replaced_img = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            replaced_img[y, x] = color_mapping[tuple(img[y, x])]

    replaced_img = cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB)
    return replaced_img


def reduce_colors(img, num_colors=4):
    contrast = 6.0
    brightness = 1.0
    img = cv2.addWeighted(img, contrast, img, 0, brightness)
    img = cv2.bilateralFilter(img, 15, 120, 120)
    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixel_values = lab_image.reshape((-1, 3))
    clt = MiniBatchKMeans(n_clusters=num_colors)
    labels = clt.fit_predict(pixel_values)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape(img.shape)
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    return quant


def create_color_block_image(colors, block_size=50):
    num_colors = len(colors)
    height = block_size * num_colors
    width = block_size
    color_block_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        start_y = i * block_size
        end_y = start_y + block_size
        color_block_image[start_y:end_y, :] = color

    return color_block_image


def process_images():
    saved_image_paths = []
    files = [f for f in os.listdir(input_images_folder) if f.lower().endswith(('.jpg', '.png'))]
    for filename in tqdm(files, desc="Processing Images"):
        image_path = os.path.join(input_images_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Step 1: Convert to depth image
            depth_img = convert_image_to_depth_image(image)
            depth_img = cv2.resize(depth_img, (256, 256))  # Resize to 256x256
            depth_img = cv2.normalize(depth_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            depth_img = np.uint8(depth_img)
            depth_img_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
            depth_output_path = os.path.join(depth_images_folder, f"depth-{filename}")
            cv2.imwrite(depth_output_path, depth_img_colored)
            saved_image_paths.append(depth_output_path)
            print(f"Saved depth image for {filename}")

            # Step 2: Reduce colors in the depth image
            reduced_depth_image = reduce_colors(depth_img_colored)
            reduced_depth_output_path = os.path.join(reduced_depth_images_folder, f"reduced-{filename}")
            cv2.imwrite(reduced_depth_output_path, reduced_depth_image)
            print(f"Saved reduced depth image for {filename}")

            # Step 3: Get pixel coordinates by color
            color_coordinates = get_pixel_coordinates_by_color(reduced_depth_image)

            # Step 4: Get average colors from the original image
            original_img_colors = [
                get_average_color(image, color_coordinates['c1'], filename, 'c1'),
                get_average_color(image, color_coordinates['c2'], filename, 'c2'),
                get_average_color(image, color_coordinates['c3'], filename, 'c3'),
                get_average_color(image, color_coordinates['c4'], filename, 'c4'),
            ]

            # Create and save the average colors block image
            average_colors_image = create_color_block_image(original_img_colors)
            average_colors_output_path = os.path.join(average_colors_folder, f"average-colors-{filename}")
            cv2.imwrite(average_colors_output_path, average_colors_image)
            print(f"Saved average colors image for {filename}")

            # Step 5: Extract and replace colors
            traced_image = extract_and_replace_colors(reduced_depth_image, original_img_colors)
            traced_output_path = os.path.join(traced_images_folder, f"traced-{filename}")
            cv2.imwrite(traced_output_path, traced_image)
            print(f"Saved traced image for {filename}")
        else:
            print(f"Failed to load {filename}")

    return saved_image_paths


process_images()
