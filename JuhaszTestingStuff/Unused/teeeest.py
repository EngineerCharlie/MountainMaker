
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import cv2
import matplotlib.pyplot as plt
import open3d as o3d

import os
import random
from pathlib import Path

import JuhaszTestingStuff.Unused.ThreeDUtils as ThreeDUtils
import JuhaszTestingStuff.Unused.Depth as Depth
import time

output_path = "C:/WorkingSets/3D/Clouds/"
os.makedirs(output_path, exist_ok=True)

for i in range(Depth.num_samples):
    cloud = ThreeDUtils.create_point_cloud(Depth.samples[i][1], Depth.samples[i][0])
    o3d.io.write_point_cloud(output_path + f"point_cloud_{i}.ply", cloud)




i = 0
depth_image = Depth.samples[i][1]
color_image = Depth.samples[i][0]

width, height = depth_image.shape

depth_image = (depth_image * 255 / np.max((depth_image))).astype('uint8')
color_image = cv2.resize(color_image, (width, height))

depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(color_image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)


custom_intrinsic = ThreeDUtils.get_intrinsics(height, width, 100) 
fx, fy = custom_intrinsic[0, 0], custom_intrinsic[1, 1]
cx, cy = custom_intrinsic[0, 2], custom_intrinsic[1, 2]
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
print("shape: ", np.asarray(pcd.points).shape)


def CopyPCD(pcd):
    new_points = np.copy(np.asarray(pcd.points))
    new_colors = np.copy(np.asarray(pcd.colors))
    #reflected_points = np.copy(original_points)
    #reflected_points[:,2] *= -1

    #original_colors = np.asarray(pcd.colors)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    return new_pcd


pcd2 = CopyPCD(pcd)
rotation = pcd2.get_rotation_matrix_from_xyz((0, .5 * np.pi, 0))
pcd2 = pcd2.rotate(rotation, center=(0,0,0))

pcd3 = CopyPCD(pcd)
rotation = pcd3.get_rotation_matrix_from_xyz((0, 1 * np.pi, 0))
pcd3 = pcd3.rotate(rotation, center=(0,0,0))

pcd4 = CopyPCD(pcd)
rotation = pcd4.get_rotation_matrix_from_xyz((0, 1.5 * np.pi, 0))
pcd4 = pcd4.rotate(rotation, center=(0,0,0))


pcd_final = o3d.geometry.PointCloud()

final_points = np.concatenate((np.asarray(pcd.points), np.asarray(pcd2.points), np.asarray(pcd3.points), np.asarray(pcd4.points)), axis=0)
final_colors = np.concatenate((np.asarray(pcd.colors), np.asarray(pcd.colors), np.asarray(pcd.colors), np.asarray(pcd.colors)), axis=0)
pcd_final.points = o3d.utility.Vector3dVector(final_points)
pcd_final.colors = o3d.utility.Vector3dVector(final_colors)


#print("points:", reflected_points[2,:])

#reflected_points[:,0] *= -1
#print("first point og: ", original_points[0,:])
#print("first point: ", reflected_points[0,:])
#test = o3d.utility.Vector3dVector(reflected_points)
#pcd_raw.points.extend(test)
#pcd_raw.colors.extend(pcd_raw.colors)
#pcd.points.append(np.array([3e-06, -1.25e-05, 1e-05]))
#pcd.colors.append(np.array([0, 0, 255]))


#print(pcd.get_center)
#pcd_raw.colors.ext(np.array([0, 0, 255]))
#o3d.visualization.draw_geometries([pcd_raw])

#o3d.visualization.draw_geometries([pcd_raw, create_point_cloud(depth_image, color_image, custom_intrinsic)])



previous_t = time.time()
# to add new points each dt secs.
dt = .5

add = 0

#pointToAdd = reflected_points[0,]
#pointToAdd2 = reflected_points[1,]
#print("shape1 ",pointToAdd.shape)
#print("shape2 ", pointToAdd2.shape)
#pointsToAdd = np.stack((pointToAdd, pointToAdd2), axis=-1)
#print("together:", pointsToAdd.shape)
#result_tensor = np.concatenate((pointToAdd, pointToAdd2))
#print(result_tensor)


# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=750, width=750)
# include it in the visualizer before non-blocking visualization.
vis.add_geometry(pcd_final)




# run non-blocking visualization. 
# To exit, press 'q' or click the 'x' of the window.
keep_running = True
while keep_running:
    
    if time.time() - previous_t > dt:

        #add += 1
        # Options (uncomment each to try them out):
        # 1) extend with ndarrays.
        #pcd.points.extend(np.random.rand(2, 3))
        #pcd.points.extend()

        # 2) extend with Vector3dVector instances.
        # pcd.points.extend(
        #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))
        
        # 3) other iterables, e.g
        # pcd.points.extend(np.random.rand(n_new, 3).tolist())

        #print("point shpae:", pointToAdd.shape)
        #pointToAdd[1][2] += add
        #pcd.points.append(pointToAdd)
        #print("add: ", add)
        #pcd.points.extend(o3d.utility.Vector3dVector(pointToAdd))
        #pcd.points.extend(np.asarray(pointToAdd.T))

        vis.update_geometry(pcd)
        previous_t = time.time()

    keep_running = vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
