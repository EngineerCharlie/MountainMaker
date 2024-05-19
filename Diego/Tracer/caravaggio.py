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

# Ensure output directory exists
Path(depth_images_folder).mkdir(parents=True, exist_ok=True)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("mps")


def convert_image_to_depth_image(img):
    input_tensor = processor(images=img, return_tensors="pt").to("mps")
    with torch.no_grad():
        outputs = model(**input_tensor)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return depth


def extract_and_replace_colors(img):
    # Declare color palette
    new_colors = [(22, 138, 173), (106, 153, 78), (56, 102, 65), (19, 42, 19)]

    # Ensure new_colors has exactly 4 colors
    if len(new_colors) != 4:
        raise ValueError("new_colors must contain exactly 4 colors.")

    # Get unique colors in the image
    unique_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # Ensure the image has exactly 4 unique colors
    if unique_colors.shape[0] != 4:
        raise ValueError("The input image must contain exactly 4 unique colors.")

    # Convert unique colors to L*a*b* color space for lightness comparison
    lab_colors = cv2.cvtColor(np.uint8([unique_colors]), cv2.COLOR_RGB2LAB)[0]

    # Sort colors based on lightness (L* value)
    sorted_indices = np.argsort(lab_colors[:, 0])
    sorted_colors = unique_colors[sorted_indices]

    # Create a mapping from the sorted colors to the new colors
    color_mapping = {tuple(sorted_colors[i]): new_colors[i] for i in range(4)}

    # Replace the colors in the image
    replaced_img = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            replaced_img[y, x] = color_mapping[tuple(img[y, x])]

    replaced_img = cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB)

    return replaced_img


def reduce_colors(img, num_colors=4):
    # define the contrast and brightness value
    contrast = 6.  # Contrast control ( 0 to 127)
    brightness = 1.  # Brightness control (0-100)
    # call addWeighted function. use beta = 0 to effectively only
    img = cv2.addWeighted(img, contrast, img, 0, brightness)

    img = cv2.bilateralFilter(img, 15, 120, 120)
    # Convert the image from RGB to L*a*b* color space
    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Reshape the image to a 2D array of pixels
    pixel_values = lab_image.reshape((-1, 3))
    # Apply k-means clustering
    clt = MiniBatchKMeans(n_clusters=num_colors)
    labels = clt.fit_predict(pixel_values)
    # Create the quantized image based on the cluster centers
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape(img.shape)
    # Convert the quantized image from L*a*b* back to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)

    # Recolor with the nature color palette
    final_output = extract_and_replace_colors(quant)

    return final_output


def process_images():
    saved_image_paths = []
    files = [f for f in os.listdir(input_images_folder) if f.lower().endswith(('.jpg', '.png'))]
    for filename in tqdm(files, desc="Processing Images"):
        image_path = os.path.join(input_images_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            depth_img = convert_image_to_depth_image(image)
            depth_img = cv2.normalize(depth_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            depth_img = np.uint8(depth_img)
            depth_img_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

            # Save the depth image
            depth_output_path = os.path.join(depth_images_folder, f"depth-{filename}")
            cv2.imwrite(depth_output_path, depth_img_colored)
            saved_image_paths.append(depth_output_path)
            print(f"Saved depth {filename} to {depth_output_path}")

            # Save the reduced colors image
            traced_image = reduce_colors(depth_img_colored)
            traced_output_path = os.path.join(traced_images_folder, f"traced-{filename}")
            cv2.imwrite(traced_output_path, traced_image)
            print(f"Saved traced {filename} to {traced_output_path}")

        else:
            print(f"Failed to load {filename}")

    return saved_image_paths


process_images()
