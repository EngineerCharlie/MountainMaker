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


def CopyPCD(pcd, invert=False):
    new_points = np.copy(np.asarray(pcd.points))
    new_colors = np.copy(np.asarray(pcd.colors))
    
    if(invert):
        new_points[:,2] *= -1

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    return new_pcd


def PCD_To_Mesh(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius*2]))

    o3d.visualization.draw_geometries([bpa_mesh], window_name="Mesh")


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
    
    
    geometries = []
    pcd_2 = CopyPCD(pcd_raw, invert=True)
    #pcd_3 = CopyPCD(pcd_raw)
    #rotation = pcd_3.get_rotation_matrix_from_xyz((0, 1 * np.pi, 0))
    #pcd_3 = pcd_3.rotate(rotation, center=(0,0,0))
    
    print("first point of pcd_raw:", np.asarray(pcd_raw.points)[0], " center point: ", pcd_raw.get_center())
    #pcd_raw.points = o3d.utility.Vector3dVector(np.asarray(pcd_raw.points)*100000-pcd_raw.get_center())
    print("first point of pcd_raw after elevating:", np.asarray(pcd_raw.points)[0], " center point: ", pcd_raw.get_center())
    #geometries.append(pcd_raw)
   
    #geometries.append(pcd_2)
    
    num_rotations = 1
    degree_step = 360 / (num_rotations)
    
    for i in range(num_rotations):

        pcd_i = CopyPCD(pcd_raw)
        rotation = pcd_i.get_rotation_matrix_from_xyz((0, i * degree_step * np.pi / 180, 0))
        #pcd_i = pcd_i.rotate(rotation, center=pcd_i.get_center())
        pcd_i = pcd_i.rotate(rotation, center=(0,0,0))

        geometries.append(pcd_i)
    
    import JuhaszTestingStuff.NoiseGenerator as NG
    #noise_pcd = NG.GetNoisePCD((25,25))
    #print("first point of noise_pcd:", np.asarray(noise_pcd.points)[0], " center point: ", noise_pcd.get_center())
    
    #geometries.append(noise_pcd)

    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud")


#   normalizedColors = np.asarray(pcd_raw.colors) / 255
#   pcd_raw.colors = o3d.utility.Vector3dVector(normalizedColors)
    
    #o3d.io.write_point_cloud(output_path + f"generated_cloud_{i}.ply", pcd_raw)

    #PTM.PCD_To_Mesh(pcd_raw)
    #PCD_To_Mesh(pcd_raw)



#extend the generated 2d image and thus you get a bigger sorrounding

#semantic segmentation
#NERF
#inverse rendering mitsuba 3
    #shape optimization
    #not with pytorch but compatible
