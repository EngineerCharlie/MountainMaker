
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

import open3d as o3d
import numpy as np


def CreateNoiseMap2D(size, octaves=3, seed=1):
    noiseMap = PerlinNoise(octaves=octaves, seed=seed)
    xpix, ypix = size[0], size[1]

    #2D
    noise = np.asarray([[noiseMap([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)])
    #plt.imshow(noise, cmap='gray')
    #plt.show()
    return noise

def CreateNoiseMap3D(size, octaves=3, seed=1):
    noiseMap = PerlinNoise(octaves=octaves, seed=seed)
    xpix, ypix, zpix = size[0], size[1], size[2]

    #3D
    noise = np.asarray([[[noiseMap([i/xpix, j/ypix, k/zpix]) for j in range(xpix)] for i in range(ypix)] for k in range(zpix)], dtype=np.float64)
    return noise

def GetColorArray(noisemap, thresholds, colors):

    noisemap = noisemap.reshape(noisemap.shape[0]**2, 1)
    color_array = []

    # Define colors based on the range of values along axis 2
    for i in range(noisemap.shape[0]):
        if noisemap[i] <= thresholds[0]:
            color_array.append(colors[0]) #[31/255, 161/255, 244/255])  # Water
        elif noisemap[i] <= thresholds[1]:
            color_array.append(colors[1])   # Green
        elif noisemap[i] <= thresholds[2]:    	
            color_array.append(colors[2])  # Mountain
        else:
            color_array.append(colors[3])   # Snow

    return color_array

thresholds_1 = [0.005, 0.3, 0.5]
colors_1 = [[31/255, 161/255, 244/255], [56/255, 118/255, 29/255], [0.3, 0.3, 0.3], [0.9, 0.9, 0.9]]

thresholds_2 = [0.005, 0.1, 0.4]
colors_2 = [[0.9, 0.9, 0.9], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.9, 0.9, 0.9]]

def GetNoisePCD(size=(100, 100), seed=1):
    #noisemap = CreateNoiseMap2D((200, 200), octaves=5, seed=11)´
    noisemap = CreateComplexNoiseMap2D(size, seed=seed)
    #print("max of noisemap: ", np.max(noisemap))
    # Define the threshold
    #threshold = 0.25
    threshold = -1

    # Find the indices where the values are above the threshold
    tensor_2d_indices = np.argwhere(noisemap > threshold)

    #noisemap[np.where(noisemap < 0.005)] = 0.005

    Z_values = (noisemap*45).reshape(noisemap.shape[0]**2,1)

    #print("tensor_2d shape: ", tensor_2d_indices.shape)

    tensor_3d = np.hstack((tensor_2d_indices, np.full((tensor_2d_indices.shape[0], 1), Z_values)))

    #print("number of points extracted from noiseMap: ", tensor_3d.shape[0])
    #color_array = np.tile(np.array([0, 0, 0]), (tensor_3d.shape[0], 1))
    color_array = GetColorArray(noisemap, thresholds_2, colors_2)

    noise_pcd = o3d.geometry.PointCloud()
    noise_pcd.points = o3d.utility.Vector3dVector(tensor_3d)
    noise_pcd.colors = o3d.utility.Vector3dVector(color_array)

    #o3d.visualization.draw_geometries([noise_pcd])
    return noise_pcd

def GetPCDfromTextureMap(texturemap, indicy_scale=5, Y_scale=2.5, customColors=False, color_array=None):

    # Assuming you have a 2D matrix called 'texturemap'
    # Replace this with your actual data

    # Get the indices of all values in the texturemap
    tensor_2d_indices = np.indices(texturemap.shape).reshape(2, -1).T
    #tensor_2d_indices = np.indices(texturemap.shape)

    # Swap the x and y indices
    tensor_2d_indices = tensor_2d_indices[:, [1, 0]]

    #print("indices: ", tensor_2d_indices)

    # Create an array of zeros to store the 3D points
    tensor_3d = np.zeros((tensor_2d_indices.shape[0], 3))
    #print("tensor_3d shape 1:", tensor_3d.shape)
    #print(tensor_3d)

    # Fill in the first two coordinates with the indices
    #tensor_3d[:, :2] = tensor_2d_indices * indicy_scale

    # Put 2D indices into x and z positions
    tensor_3d[:, 0] = tensor_2d_indices[:, 0] * indicy_scale  # x position
    tensor_3d[:, 2] = tensor_2d_indices[:, 1] * indicy_scale  # z position

    #print("tensor_3d shape 2:", tensor_3d.shape)
    #print(tensor_3d)

    # Multiply the texturemap values by a factor (e.g., 45)
    Y_values = texturemap * Y_scale

    # Set the third coordinate with the modified texturemap values
    #tensor_3d[:, 2] = Z_values.flatten()
    tensor_3d[:, 1] = Y_values.flatten()

    #print("normal tensor shape: ", tensor_3d.shape)
    filtered_tensor_3d = tensor_3d[tensor_3d[:, 2] != 0]
    #print("filtered tensor shape: ", filtered_tensor_3d.shape)

    # Print the resulting 3D tensor
    #print("filtered_tensor_3d shape 4:", filtered_tensor_3d.shape)
    #print(filtered_tensor_3d[:25])
    
    if(customColors == False):
        color_array = np.tile(np.array([0, 0, 0]), (filtered_tensor_3d.shape[0], 1))
    else:
        #print("color shape1: " , color_array.shape)
        #color_array = color_array.flatten()
        color_array = color_array.reshape(color_array.shape[0]**2, 3)
        #print("color shape2: " , color_array.shape)
        color_array = color_array[color_array[:] != [0,0,0]]
        #print("color shape3: " , color_array.shape)

    #color_array = GetColorArray(texturemap, thresholds_2, colors_2)

    noise_pcd = o3d.geometry.PointCloud()
    noise_pcd.points = o3d.utility.Vector3dVector(filtered_tensor_3d)
    noise_pcd.colors = o3d.utility.Vector3dVector(color_array)

    #o3d.visualization.draw_geometries([noise_pcd])
    return noise_pcd

def CreateComplexNoiseMap2D(size, scale=1, seed=1, texture_map=None):
    octaves = [3, 6, 12, 24]  # Adjust these values as needed
    weights = np.array([1.0, 0.5, 0.25, 0.125])  # Adjust weights for different noise effects

    xpix, ypix = size[0], size[1]
    x = np.linspace(0, 1, xpix) / scale
    y = np.linspace(0, 1, ypix) / scale
    xv, yv = np.meshgrid(x, y, indexing='ij')

    noisemap = np.zeros((xpix, ypix))

    for octave, weight in zip(octaves, weights):
          noise = PerlinNoise(octaves=octave, seed=seed)
          noise_map = np.zeros((xpix, ypix))
          for i in range(xpix):
              for j in range(ypix):
                  noise_map[i, j] = noise([xv[i, j], yv[i, j]])
          noisemap += weight * noise_map

    if texture_map is not None:
      noisemap *= texture_map

    return noisemap

def create_gradient_circle(size):
    xpix, ypix = size
    xv, yv = np.meshgrid(np.linspace(-1, 1, xpix), np.linspace(-1, 1, ypix), indexing='ij')
    gradient_circle = np.sqrt(xv**2 + yv**2)
    gradient_circle = np.clip(1 - gradient_circle, 0, 1)
    return gradient_circle


#import PointCloudToMesh.PointCloudToMesh as PTM
#PTM.PCD_To_Mesh(noise_pcd)
if __name__ == "__main__":
    #noise_pcd = GetNoisePCD((100, 100))
    #o3d.visualization.draw_geometries([noise_pcd])

    pass



