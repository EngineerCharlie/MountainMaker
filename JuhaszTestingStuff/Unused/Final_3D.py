import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import cv2
import matplotlib.pyplot as plt
import open3d as o3d

import os
import random
#from pathlib import Path

#import ThreeDUtils
#import Depth
#import time
import JuhaszTestingStuff.PointCloudToMesh as PTM

input_path = "C:/WorkingSets/3D/OriginalImages/"
output_path = "C:/WorkingSets/3D/Clouds/"


#num_samples = 3
#selected_images = random.sample(os.listdir(input_path), num_samples)

num_samples = 1
selected_images = ['test.jpg']

print("Images sampled: ", selected_images)
images = []

for i in range(num_samples):
    original_img = cv2.imread((input_path + selected_images[i]))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    images.append(original_img)

    #print("Sampled image shape: ", original_img.shape)


processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("cuda")
#print(model.config)

#pre process images
samples = []
for i in range(num_samples):
    input = processor(images = images[i], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model(**input)
        depth = output.predicted_depth

    depth = depth.squeeze().cpu().numpy()

    samples.append([images[i], depth])

'''
for i in range(num_samples):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(samples[i][0])
    axs[1].imshow(samples[i][1])
    plt.show()
'''

'''
def get_intrinsics(H, W, fov=55.0):

    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1 ]])

def pixel_to_point(depth_image, camera_intrinsics = None):

    height, width = depth_image.shape[0], depth_image.shape[1]
    if(camera_intrinsics is None):
        camera_intrinsics = get_intrinsics(height, width, fov=55.0)
    
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    u, v = np.meshgrid(x, y)

    x_over_z = (u - cx) / (fx)
    y_over_z = (v - cy) / (fy)

    z = depth_image / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    return x, y, z


def create_point_cloud(depth_image, color_image, camera_intrinsics = None, scale_ratio=100.0):

    print("SHAPE: ", depth_image.shape)
    height, width = depth_image.shape[0], depth_image.shape[1]
    if(camera_intrinsics is None):
        camera_intrinsics = get_intrinsics(height, width, fov=55.0)

    color_image = cv2.resize(color_image, (height, width))

    depth_image = np.maximum(depth_image, 1e-5)

    depth_image = scale_ratio / depth_image

    #depth_image = (depth_image * 255/ np.max(depth_image)).astype('uint8')

    x, y, z = pixel_to_point(depth_image, camera_intrinsics)
    point_image = np.stack((x, y, z), axis=-1)
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_image.reshape(-1, 3))
    cloud.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3)/255.0)

    return cloud
'''


#os.makedirs(output_path, exist_ok=True)

#for i in range(num_samples):
#    cloud = create_point_cloud(samples[i][0], samples[i][1])
#    o3d.io.write_point_cloud(output_path + f"generated_cloud_{i}.ply", cloud)


for i in range(num_samples):
    depth_image = samples[i][1]
    color_image = samples[i][0]

    width, height = depth_image.shape[0], depth_image.shape[1]
    depth_image = (depth_image * 255/ np.max(depth_image)).astype('uint8')
    color_image = cv2.resize(color_image, (height, width))

    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 400, 400, width/2, height/2)

    pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    o3d.visualization.draw_geometries([pcd_raw])

    #PTM.PCD_To_Mesh(pcd_raw)

