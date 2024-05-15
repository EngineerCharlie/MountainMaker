import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

# Paths for image input and output
valid_images_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/academia"
drawing_images_folder = "/Users/diegovelotto/Documents/GitHub/MountainMaker/Diego/renaissance/atelier"

# Ensure output directory exists
Path(drawing_images_folder).mkdir(parents=True, exist_ok=True)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("mps")


def convert_image_to_depth_image(img):
    input_tensor = processor(images=img, return_tensors="pt").to("mps")
    with torch.no_grad():
        outputs = model(**input_tensor)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return depth


def reduce_colors(img, num_colors=6):
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
    return quant


def process_images():
    saved_image_paths = []
    files = [f for f in os.listdir(valid_images_folder) if f.lower().endswith(('.jpg', '.png'))]
    for filename in tqdm(files, desc="Processing Images"):
        image_path = os.path.join(valid_images_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            depth_img = convert_image_to_depth_image(image)
            depth_img = cv2.normalize(depth_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            depth_img = np.uint8(depth_img)
            depth_img_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

            # Save the depth image
            depth_output_path = os.path.join(drawing_images_folder, f"depth-{filename}")
            cv2.imwrite(depth_output_path, depth_img_colored)
            saved_image_paths.append(depth_output_path)
            print(f"Saved depth {filename} to {depth_output_path}")
        else:
            print(f"Failed to load {filename}")

    return saved_image_paths


process_images()
