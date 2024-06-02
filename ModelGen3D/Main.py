import matplotlib as plt


# from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

import open3d as o3d
import numpy as np
import torch

import open3d as o3d
import ModelGen3D.NoiseGenerator as NG
import ModelGen3D.CustomUtility as CU
import ModelGen3D.ImageToPCD as F3D

# import ModelGen3D.InferBoundless as IB

import cv2

# from skimage import data
# from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint


def generate_3d_model():
    print("Generating 3D model")
    input_path = ""
    filename = "generated_image.jpg"

    original_img = cv2.imread((input_path + filename))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # original_img = IB.CreateInferModelInferation((input_path + filename), 2)
    # original_img = np.asarray(original_img*255, dtype=np.uint8)

    geom = F3D.GET_PCD(original_img, showGraphics=True)
    orig_pcd = geom[0]

    points = np.asarray(orig_pcd.points)
    reference_point = points[0]
    # print("Reference Point: " , reference_point)
    # print("original point count: ", len(points), " in the shape of: ", points.shape)
    # print(points[:5])

    down_sampled_pcd = orig_pcd.voxel_down_sample(voxel_size=0.00005)
    ds_points = (
        np.asarray(down_sampled_pcd.points) - down_sampled_pcd.get_center()
    ) * 10**5
    down_sampled_pcd.points = o3d.utility.Vector3dVector(ds_points)
    # print("down_sampled point count: ", len(ds_points), " in the shape of: ", ds_points.shape)
    # print(ds_points[:5])

    # fixing original pcd coords too
    orig_pcd.points = o3d.utility.Vector3dVector(
        (np.asarray(orig_pcd.points) - orig_pcd.get_center()) * 10**5
    )

    # o3d.visualization.draw_geometries([orig_pcd], window_name="Point Cloud")
    o3d.visualization.draw_geometries(
        [down_sampled_pcd], window_name="Point Cloud down_sampled"
    )

    heightmap = CU.Project_PCD_TO_MAP(down_sampled_pcd)
    plt.imsave(f"heightmap_{filename}", heightmap)

    num_rows = heightmap.shape[0]
    # heightmap = np.flipud(heightmap)
    size = (int(num_rows * 3), int(num_rows * 3))

    fig, axs = plt.subplots(3, 3)

    # print("HEIGHT VALUES: ")
    # print(heightmap[:10])

    # Plot the heightmap
    axs[0, 0].imshow(heightmap, cmap="gray", origin="lower")
    axs[0, 0].set_title("projected heightmap")
    axs[0, 0].axis("off")

    # axs[0][0].colorbar(label='Height (z)')
    # axs[0, 0].title('2D Heightmap from 3D Point Cloud')
    # plt.show()

    enlarged_heightmap = np.full(size, 0)
    # enlarged_heightmap[:num_rows, :num_rows] = heightmap
    enlarged_heightmap[size[0] - num_rows : size[0], :num_rows] = heightmap
    axs[0, 1].imshow(enlarged_heightmap, cmap="gray", origin="lower")
    axs[0, 1].set_title("enlarged heightmap")
    axs[0, 1].axis("off")

    import random

    noiseMap = NG.CreateComplexNoiseMap2D(size, 3, random.randint(0, 10000))
    # noiseMap = NG.CreateComplexNoiseMap2D(size, 3, 10)

    noiseMap *= 10**2
    int_noise_map = np.round(noiseMap).astype(int)
    axs[0, 2].imshow(int_noise_map, cmap="gray", origin="lower")
    axs[0, 2].set_title("complex perlin noise")
    axs[0, 2].axis("off")
    # axs[0][1].colorbar(label='Height (z)')
    # axs[0, 1].title('Noise Map to apply')
    # plt.show()

    # print("NOISE VALUES: ")
    # print(int_noise_map[:10])

    noisy_heightmap_blocky = np.copy(int_noise_map)
    noisy_heightmap = np.copy(int_noise_map)

    # Overlay the smaller matrix values
    # noisy_heightmap[:num_rows, :num_rows] = heightmap
    noisy_heightmap_blocky[size[0] - num_rows : size[0], :num_rows] = heightmap
    noisy_heightmap = np.where(
        enlarged_heightmap != 0, enlarged_heightmap, noisy_heightmap
    )

    axs[1, 0].imshow(noisy_heightmap_blocky, cmap="gray", origin="lower")
    # plt.colorbar(label='Height (z)')
    axs[1, 0].set_title("blocky: noise + heightmap")
    axs[1, 0].axis("off")
    # plt.show()

    # Identify black pixels (value 0)
    mask = noisy_heightmap_blocky == 0

    # Flip the mask vertically before inpainting
    # mask = np.flipud(mask)

    # mask should be a random noise?

    # Inpaint using biharmonic method
    # image_inpainted = inpaint_biharmonic(heightmap, mask, multichannel=False)
    # print("heightmap shape: ", heightmap.shape, " mask shape: ", mask.shape, " noise heightmap shape: ", noisy_heightmap_blocky)
    image_result = inpaint.inpaint_biharmonic(noisy_heightmap_blocky, mask)

    image_result_flipped = np.flipud(image_result)

    # image_final = np.where(heightmap == 0, image_result, heightmap)

    # Display the result
    axs[1, 1].imshow(image_result_flipped, cmap="gray")
    # plt.colorbar()
    axs[1, 1].set_title("biharmonic smoothing applied")
    axs[1, 1].axis("off")
    # plt.show()

    heightmap_estimate = np.where(enlarged_heightmap == 0, image_result, 0)

    # Display the result
    axs[1, 2].imshow(heightmap_estimate, cmap="gray")
    axs[1, 2].set_title("heightmap estimate")
    axs[1, 2].axis("off")
    # plt.colorbar()
    # plt.title('Estimate heightmap')
    # plt.show()

    axs[2, 0].imshow(noisy_heightmap, cmap="gray")
    axs[2, 0].set_title("sampled: noise + heightmap")
    axs[2, 0].axis("off")
    # plt.show()

    # image_result2 = cv2.GaussianBlur(noisy_heightmap, ksize=(3, 3), sigmaX=3)
    blured_noisy_heightmap = cv2.blur(noisy_heightmap, ksize=(5, 5))

    axs[2, 1].imshow(blured_noisy_heightmap, cmap="gray")
    # plt.colorbar()
    axs[2, 1].set_title("Blurred: noise + heightmap")
    axs[2, 1].axis("off")
    # plt.show()

    solution = np.where(
        enlarged_heightmap != 0, enlarged_heightmap, blured_noisy_heightmap
    )

    axs[2, 2].imshow(solution, cmap="gray")
    axs[2, 2].set_title("blurred noise + original heightmap")
    axs[2, 2].axis("off")

    plt.show()

    heightmap_estimate[:] = heightmap_estimate[:] * 10**9

    projected_pcd = NG.GetPCDfromTextureMap(
        heightmap, indicy_scale=5, Y_scale=1.25, customColors=False
    )

    projected_estimation_pcd = NG.GetPCDfromTextureMap(
        heightmap_estimate, indicy_scale=5, Y_scale=2.5, customColors=False
    )
    # projected_entire_pcd = NG.GetPCDfromTextureMap(image_result_flipped, indicy_scale=5, Z_scale=2, customColors=False)

    # projected_pcd_points = np.asarray(projected_pcd.points)
    # projected_pcd.points = o3d.utility.Vector3dVector(projected_pcd_points-(projected_pcd_points[0]-reference_point))
    # projected_pcd_points = np.asarray(projected_pcd.points)

    # projected_estimation_pcd.points = o3d.utility.Vector3dVector(np.asarray(projected_estimation_pcd.points)-down_sampled_pcd.get_center())
    projected_estimation_pcd_points = np.asarray(projected_estimation_pcd.points)
    projected_estimation_pcd_points[:, 0] = (
        projected_estimation_pcd_points[:, 0] - reference_point[0]
    )
    projected_estimation_pcd_points[:, 2] = (
        projected_estimation_pcd_points[:, 2] - reference_point[2]
    )
    projected_estimation_pcd.points = o3d.utility.Vector3dVector(
        projected_estimation_pcd_points
    )
    projected_estimation_pcd_points = np.asarray(projected_estimation_pcd.points)

    orig_x_coords = points[:, 0]
    orig_z_coords = points[:, 2]

    width_of_translation = reference_point[0] - np.min(orig_x_coords)

    sheet_z_coords = projected_estimation_pcd_points[:, 2]
    depth_of_sheet = np.max(sheet_z_coords) - np.min(sheet_z_coords)

    depth_of_translation = np.max(orig_z_coords) - np.min(sheet_z_coords)

    # print("width_of_translation: ", width_of_translation)
    # print("depth_of_translation: ", depth_of_translation)
    # print("depth_of_sheet: ", depth_of_sheet)
    projected_estimation_pcd_points[:, 0] = (
        projected_estimation_pcd_points[:, 0] - width_of_translation
    )
    projected_estimation_pcd_points[:, 2] = (
        projected_estimation_pcd_points[:, 2] + depth_of_translation - depth_of_sheet
    )
    projected_estimation_pcd.points = o3d.utility.Vector3dVector(
        projected_estimation_pcd_points
    )

    # o3d.visualization.draw_geometries([projected_pcd, projected_estimation_pcd, down_sampled_pcd, projected_entire_pcd])
    # o3d.visualization.draw_geometries([projected_pcd, projected_estimation_pcd, down_sampled_pcd, orig_pcd])
    # o3d.visualization.draw_geometries([projected_estimation_pcd, orig_pcd])
    o3d.visualization.draw_geometries([projected_estimation_pcd, orig_pcd])

    # o3d.visualization.draw_geometries([orig_pcd])

    # If you want mesh
    # import PointCloudToMesh as PCTM

    # PCTM.PCD_To_Mesh(orig_pcd)
