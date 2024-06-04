import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def read_image(image_path):
    """Read an image from a file."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image


def resize_to_300ppi(image, current_ppi=72):
    """Resize the image to 300 PPI from the current PPI."""
    # Get the current dimensions of the image
    height, width = image.shape[:2]
    # Calculate the scaling factor
    scale_factor = 300 / current_ppi
    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_image


def improve_lighting(image):
    """Improve the lighting of the image using histogram equalization."""
    # Convert the image to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Split the channels
    y, cr, cb = cv2.split(ycrcb)
    # Apply histogram equalization on the Y channel
    y_eq = cv2.equalizeHist(y)
    # Merge the channels back
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    # Convert back to BGR color space
    image_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return image_eq


def denoise_image(image):
    """Denoise the image using Non-Local Means Denoising."""
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
    return denoised_image


def adjust_contrast(image, alpha=0.9, beta=0):
    # Adjust contrast and brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def adjust_saturation(image, saturation_scale=1.8):
    """Adjust the saturation of the image."""
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Scale the saturation channel
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, saturation_scale)
    s = np.clip(s, 0, 255).astype(hsv.dtype)
    # Merge the channels back
    hsv = cv2.merge([h, s, v])
    # Convert back to BGR color space
    saturated_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return saturated_image


def lens_blur(image, ksize=8):
    """Apply a lens blur (bokeh effect) to the image."""
    # Create a circular kernel
    kernel = np.zeros((ksize, ksize), np.float32)
    center = ksize // 2
    cv2.circle(kernel, (center, center), center, 1, -1)
    kernel = kernel / np.sum(kernel)

    # Apply the filter2D function to blur the image with the circular kernel
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def save_image(image, output_path):
    """Save the processed image to a file."""
    success = cv2.imwrite(output_path, image)
    if not success:
        print(f"Failed to save image: {output_path}")
    else:
        print(f"Image successfully saved: {output_path}")


def process_image(image_path, output_folder):
    """Process an image: read, resize to 300 PPI, improve lighting, denoise, sharpen, adjust saturation, and save."""
    print(f"Processing image: {image_path}")
    # Read the image
    image = read_image(image_path)
    # Resize to 300 PPI
    resized_image = resize_to_300ppi(image)
    # Improve lighting
    image_eq = improve_lighting(resized_image)
    # Denoise the image
    denoised_image = denoise_image(image_eq)
    # Adjust saturation
    saturated_image = adjust_saturation(denoised_image)
    # Adjust contrast
    contrasted_image = adjust_contrast(saturated_image)
    # Apply lens blur
    final_img = lens_blur(contrasted_image)
    # Prepare the output path
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    print(f"Saving processed image to: {output_path}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)  # Display input image
    axes[0].set_title("Generated Image")
    axes[0].axis("off")
    axes[1].imshow(final_img)  # Display output image
    axes[1].set_title("Processed Image")
    axes[1].axis("off")
    plt.show()
    # Save the processed image
    save_image(final_img, output_path)


def batch_process_images(input_folder, output_folder):
    """Process all images in the input folder and save to the output folder."""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder)


# Example usage
def __main__():
    input_folder = '/Users/diegovelotto/Documents/GitHub/MountainMaker/pre-processing-test/inp'
    output_folder = '/Users/diegovelotto/Documents/GitHub/MountainMaker/pre-processing-test/out'
    batch_process_images(input_folder, output_folder)
