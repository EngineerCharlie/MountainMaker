
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
from cv2.typing import MatLike, Point, Size
import numpy as np
from Processer import PostProcesser
import os




#original_images_folder = "C:\Users\Juhász\NightSkyify\PytorchGAN\TrainingSet\Original"
#original_images_folder = "C:\\Users\\Juhász\\NightSkyify\\PytorchGAN\\TrainingSet\\Original"
#original_images_folder = "TrainingSet/Original"

#drawing_images_folder = "C:\Users\Juhász\NightSkyify\Pytorch GAN\TrainingSet\Drawing"
#drawing_images_folder = "C:\\Users\\Juhász\\NightSkyify\\PytorchGAN\\TrainingSet\\Drawing"


#test = cv2.imread("/Original/Mountain-4.jpg", -1)
#cv2.resize(test, (200,200))


'''
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

print("SCRIPT DIR: ", script_dir)
# Define relative paths for input and output folders
original_images_folder_rel = "TrainingSet/Original"
drawing_images_folder_rel = "ProcessedImages"

# Construct full paths based on the script directory
original_images_folder = os.path.join(script_dir, original_images_folder_rel)
drawing_images_folder = os.path.join(script_dir, drawing_images_folder_rel)

# Loop through each file in the input folder
for filename in os.listdir(original_images_folder):
    # Check if the file is an image (you can add more image formats if needed)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the image path
        image_path = os.path.join(original_images_folder, filename)
        print("PATH FOUND:", image_path)
        #print("encode decode:", image_path.encode('utf-8').decode('utf-8'))
        
        # Load the image using cv2.imread
        image = cv2.imread(image_path.encode('utf-8').decode('utf-8'))

        # Check if the image is loaded successfully
        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, (250, 250))

            # Process the image as needed (replace this with your actual processing logic)
            processed_img = Image.fromarray(GetProcessedImg(resized_image))

            # Construct the output path
            output_path = os.path.join(drawing_images_folder, f"Processed_{filename}")

            # Save the processed image to the output folder
            cv2.imwrite(output_path, processed_img)

            print(f"Saved {filename} to {output_path}")
        else:
            print(f"Failed to load {filename}")


#image = PostProcesser.GetImageFromPath("C:\\mountain1.jpg")
#processedImg = Image.fromarray(GetProcessedImg(image))
#processedImg.save('C:/Users/Juhász/NightSkyify/Pytorch GAN/TrainingSet/Drawing/image.png')

'''


valid_images_folder = "C:/WorkingSets/TrainingSets/Valid"
drawing_images_folder = "C:/WorkingSets/TrainingSets/Drawing"

# Loop through each file in the input folder
for filename in os.listdir(valid_images_folder):
    # Check if the file is an image (you can add more image formats if needed)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(valid_images_folder, filename)

        # Load the image using cv2.imread
        #image = PostProcesser.GetImageFromPath(image_path)
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully
        if image is not None:

            #image = cv2.resize(image, (250,250))
            # Process the image as needed
            processed_img = PostProcesser.GetProcessedImg(image)
            
            # Construct the output path using raw string literals or forward slashes
            output_path = os.path.join(drawing_images_folder, f"Processed_{filename}")
            
            # Save the processed image to the output folder
            cv2.imwrite(output_path, processed_img)
            
            print(f"Saved {filename} to {output_path}")
        else:
            print(f"Failed to load {filename}")

