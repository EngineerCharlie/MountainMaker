
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

import open3d as o3d
import numpy as np
import torch

noiseMap = PerlinNoise(octaves=5, seed=1)
xpix, ypix, zpix = 50, 50, 50

#2D
#pic = [[noiseMap([i/xpix, j/ypix, k/zpix]) for j in range(xpix)] for i in range(ypix)]
#plt.imshow(pic, cmap='gray')
#plt.show()

#3D
noise = np.asarray([[[noiseMap([i/xpix, j/ypix, k/zpix]) for j in range(xpix)] for i in range(ypix)] for k in range(zpix)], dtype=np.float64)
print("noise shape: " , noise.shape)

print(noise)
# Define the threshold
threshold = 0.25

# Find the indices where the values are above the threshold
indices = np.argwhere(noise > threshold)

# Extract the 3D vectors
vectors_array = indices

print("number of points extracted from noiseMap: ", vectors_array.shape[0])
color_array = np.tile(np.array([0, 0, 0]), (vectors_array.shape[0], 1))

noise_pcd = o3d.geometry.PointCloud()
noise_pcd.points = o3d.utility.Vector3dVector(vectors_array)
noise_pcd.colors = o3d.utility.Vector3dVector(color_array)

o3d.visualization.draw_geometries([noise_pcd])

