import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import cv2
import matplotlib.pyplot as plt
import open3d as o3d


import PointCloudToMesh as PTM

input_path = "C:/WorkingSets/3D/OriginalImages/"
output_path = "C:/WorkingSets/3D/Clouds/"


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

def GET_PCD(original_img, showGraphics=False):

    #print("Images sampled: ", selected_image)
    images = []

    images.append(original_img)

    #print("Sampled image shape: ", original_img.shape)

    processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("cuda")
    #print(model.config)

    #pre process images----------------------
    samples = []

    #images[0] = images[0][:,:256,:]
    input = processor(images = images[0], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model(**input)
        depth = output.predicted_depth

    depth = depth.squeeze().cpu().numpy()

    if(showGraphics):
        fig,axs = plt.subplots(2,1)
        axs[0].imshow(images[0])
        axs[0].axis('off')
        axs[1].imshow(depth)
        axs[1].axis('off')
        plt.show()

    samples.append(images[0])
    samples.append(depth)
    #-------------------------

    #generating 3D PCD:
    depth_image = samples[1]
    color_image = samples[0]

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
    #pcd_2 = CopyPCD(pcd_raw, invert=True)
    #pcd_3 = CopyPCD(pcd_raw)
    #rotation = pcd_3.get_rotation_matrix_from_xyz((0, 1 * np.pi, 0))
    #pcd_3 = pcd_3.rotate(rotation, center=(0,0,0))
    
    #print("first point of pcd_raw:", np.asarray(pcd_raw.points)[0], " center point: ", pcd_raw.get_center())
    #pcd_raw.points = o3d.utility.Vector3dVector(np.asarray(pcd_raw.points)*100000-pcd_raw.get_center())
    #print("first point of pcd_raw after elevating:", np.asarray(pcd_raw.points)[0], " center point: ", pcd_raw.get_center())
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
    
    
    #import NoiseGen as NG
    #noise_pcd = NG.GetNoisePCD((25,25))
    #print("first point of noise_pcd:", np.asarray(noise_pcd.points)[0], " center point: ", noise_pcd.get_center())
    
    #geometries.append(noise_pcd)

    if(showGraphics):
        o3d.visualization.draw_geometries(geometries, window_name="Point Cloud")

    return geometries
