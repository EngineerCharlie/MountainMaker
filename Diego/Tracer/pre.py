import os
from PIL import Image
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


def file_to_svg(input_filename: str):
    try:

        original_image = cv2.cvtColor(cv2.imread(input_filename), cv2.COLOR_BGR2RGB)
        print(f"Processing {input_filename}...please wait haha")
    except IOError:
        print(f"Image ({input_filename}) could not be loaded.")
        return

    def show_image(img, title):
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    # Display original image
    show_image(original_image, 'Original Image')

    def image_filler(img):
        # Apply threshold to the image to create a binary image (inversion used here)
        _, im_th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        # Copy the threshold image for flood filling
        im_floodfill = im_th.copy()

        # Get the dimensions of the image to prepare the mask
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask needs to be 2 pixels larger than the image

        # Apply floodFill - fills all 0 values that are connected to the border with white (255, 255, 255)
        cv2.floodFill(im_floodfill, mask, (0, 0), (255, 255, 255))

        # Set source images for blending
        src1 = img
        src2 = im_floodfill

        # Set the blending factor
        alpha = 0.5
        beta = 1.0 - alpha  # Beta is complementary to alpha

        # Blend images
        filled_img = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
        return filled_img

    # Apply image filters:
    def image_filters(img, num_colors=2):

        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Adaptive Histogram Equalization to the low contrast image:
        adjusted_img = exposure.equalize_adapthist(img_gray, clip_limit=0.03)

        # Calculate mean pixel value and contrast
        mean_pixel = np.mean(img_gray)
        contrast = np.std(img_gray)

        # Apply gamma correction for low contrast and high brightness
        if contrast < 80 < mean_pixel:
            gamma = 1.2  # Gamma level for correction
            img_float = img_gray.astype(np.float32) / 255.0  # Convert to float32 for gamma correction
            adjusted_img = np.power(img_float, gamma)  # Apply gamma correction
            adjusted_img = (adjusted_img * 255).astype(np.uint8)  # Scale back to 0-255 range
        else:
            adjusted_img = img_gray

        show_image(adjusted_img, ('Adjusted Image'))

        # Apply Gaussian blur for smoothness
        adjusted_img = cv2.GaussianBlur(adjusted_img, (7, 7), 0)

        # Color quantization using k-means clustering
        pixels = adjusted_img.reshape(-1, 1)
        pixels = np.float32(pixels)  # Convert pixels to float32 for k-means

        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers to uint8 and flatten labels to recreate image
        centers = np.uint8(centers)
        quantized_img = centers[labels.flatten()]
        quantized_img = quantized_img.reshape(adjusted_img.shape)  # Reshape to original image dimensions

        # Convert back to BGR for output
        adjusted_img = cv2.cvtColor(quantized_img, cv2.COLOR_GRAY2BGR)

        return adjusted_img

    def image_to_black_and_white(img):
        # Filter out the image:
        img = image_filters(img)

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to binary (black and white)
        _, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Get the unique colors (black and white)
        unique_colors, counts = np.unique(binary_img, return_counts=True)

        # Check if there are exactly two unique colors
        if len(unique_colors) != 2:
            print(f"Error: The image does not contain exactly two unique colors. It contains: {len(unique_colors)}")
            return None

        # Determine the colors
        black = min(unique_colors)
        white = max(unique_colors)

        # Create a mask for each unique color
        mask1 = (binary_img == black)
        mask2 = (binary_img == white)

        # Create a new image and set the colors based on the masks
        new_img = np.zeros_like(img)
        new_img[mask1] = [0, 0, 0]  # black
        new_img[mask2] = [255, 255, 255]  # white

        return new_img

    # Display the processed image after filtering
    filtered_img = image_to_black_and_white(original_image)
    show_image(filtered_img, 'Filtered Image')

    # Fill the holes of the images
    filled_img = image_filler(filtered_img)
    show_image(filled_img, 'Filled Image')

    # Change the colors of the image to black and white again
    processed_img = image_to_black_and_white(filled_img)
    show_image(processed_img, 'Final Processed Image')

    # Save processed image
    output_folder = "processed"  # Output folder
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists

    processed_img = Image.fromarray(processed_img)
    processed_img.save(os.path.join(output_folder, os.path.basename(input_filename)))

    # Adjust blacklevel and turnpolicy for better tracing
    bm = Bitmap(processed_img, blacklevel=0.4)  # Adjusted blacklevel
    plist = bm.trace(turnpolicy=POTRACE_TURNPOLICY_MINORITY, alphamax=0.5, opttolerance=0.2)

    output_folder = "output"  # Output folder for SVG
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists

    output_filename = os.path.splitext(os.path.basename(input_filename))[0] + "-output.svg"
    output_path = os.path.join(output_folder, output_filename)  # Path for the output file

    with open(output_path, "w") as fp:
        fp.write(
            f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{processed_img.width}" height="{processed_img.height}" viewBox="0 0 {processed_img.width} {processed_img.height}">'
        )

        for i, curve in enumerate(plist):
            parts = []
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")

            for segment in curve:
                if segment.is_corner:
                    a = segment.c
                    b = segment.end_point
                    parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                else:
                    c1 = segment.c1
                    c2 = segment.c2
                    end = segment.end_point
                    parts.append(f"C{c1.x},{c1.y} {c2.x},{c2.y} {end.x},{end.y}")

            '''# Get average color for the current curve
            color = get_average_color(original_image, curve)
            color_hex = "#{:02x}{:02x}{:02x}".format(*color)'''

            fp.write(
                f'<path stroke="none" fill="000000" fill-rule="nonzero" d="{"".join(parts)}"/>'
            )

        fp.write("</svg>")
    print(f"SVG file saved as {output_path}")


if __name__ == '__main__':
    input_folder = 'input'
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg')):
            file_path = os.path.join(input_folder, file)
            file_to_svg(file_path)
