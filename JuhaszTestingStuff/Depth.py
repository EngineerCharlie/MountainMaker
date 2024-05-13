import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import cv2
import matplotlib.pyplot as plt
import open3d as o3d

import os
import random
from pathlib import Path

images = []
num_samples = 1


pathToImage = "C:/WorkingSets/3D/Images/0.jpg"
input_img = cv2.imread(pathToImage)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
images.append(input_img)


'''
for i in range(num_samples):
    pathToImage = "C:/WorkingSets/3D/Images/"+str(i)+".jpg"
    input_img = cv2.imread(pathToImage)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    #print(input_img.shape)

    images.append(input_img)
'''


processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("cuda")

#print(model.config)

samples=[]
for i in range(num_samples):
    input = processor(images=images[i], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**input)
        depth = outputs.predicted_depth
    
    depth = depth.squeeze().cpu().numpy()

    samples.append([images[i], depth])

def plot_samples():
    for i in range(num_samples):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(samples[i][0])
        axs[0].set_title("Original Image")
        axs[1].imshow(samples[i][1])
        axs[1].set_title("Depth Image")
        plt.show()

#plot_samples()