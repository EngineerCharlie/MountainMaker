import cv2
import os
from Processer import PostProcesser

original_images_folder = "C:/TrainingSet/Original"
valid_images_folder = "C:/TrainingSet/Valid"

# Loop through each file in the input folder
for filename in os.listdir(original_images_folder):
    # Check if the file is an image (you can add more image formats if needed)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(original_images_folder, filename)

        # Load the image using cv2.imread
        #image = PostProcesser.GetImageFromPath(image_path)
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if image is not None and image.shape[0] >= 250 and image.shape[1] >= 250:

            image = cv2.resize(image, (250,250))
            # Process the image as needed
            #processed_img = PostProcesser.GetProcessedImg(image)
            
            # Construct the output path using raw string literals or forward slashes
            output_path = os.path.join(valid_images_folder, f"Valid_{filename}")
            
            # Save the processed image to the output folder
            cv2.imwrite(output_path, image)
            
            print(f"Saved {filename} to {output_path}")
        else:
            print(f"Failed to load {filename}")