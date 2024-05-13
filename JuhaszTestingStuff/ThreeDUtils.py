import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import cv2
import matplotlib.pyplot as plt
import open3d as o3d

import os
import random
from pathlib import Path


def get_intrinsics(H, W, fov = 55.0):
    '''
    Intrinsics for a pinhole camera model
    Assume fov of 55 degrees and central principal point
    '''

    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def pixel_to_point (depth_image, camera_intrinsics=None):
    '''
    Convert depth image to 3D points
    Assume fov of 55 degrees
    '''

    height, width = depth_image.shape
    if (camera_intrinsics is None):
        camera_intrinsics = get_intrinsics(height, width, fov=55.0)
    
    #Create u, v meshgrid and precompute projection triangle ratios
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height -1, height)
    u, v = np.meshgrid(x, y)

    x_over_z = (u - cx) / (fx)
    y_over_z = (v - cy) / (fy)

    #3-D pythagoras re-arranged for z
    z = depth_image / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    
    return x, y, z

def create_point_cloud(depth_image, color_image, camera_intrinsics=None, scale_ratio=100.0):

    height, width = depth_image.shape
    if (camera_intrinsics is None):
        camera_intrinsics = get_intrinsics(height, width, fov=55.0)
    
    color_image = cv2.resize(color_image, (width, height))

    #make sure that depth_image does not contain any zeros
    depth_image = np.maximum(depth_image, 1e-5)

    depth_image = scale_ratio / depth_image

    x, y, z = pixel_to_point(depth_image, camera_intrinsics)
    point_image = np.stack((x, y, z), axis=-1)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_image.reshape(-1, 3))
    cloud.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

    #Masking for outdoor skies
    #mask = point_image[:,:,2] < 1e3
    #cloud.points = o3d.utility.Vector3dVector(point_image[mask].reshape(-1, 3))
    #cloud.points = o3d.utility.Vector3dVector(color_image[mask].reshape(-1, 3) / 255.0)

    return cloud

def create_points(depth_image, color_image, camera_intrinsics=None, scale_ratio=100.0):

    height, width = depth_image.shape
    if (camera_intrinsics is None):
        camera_intrinsics = get_intrinsics(height, width, fov=55.0)
    
    color_image = cv2.resize(color_image, (width, height))

    #make sure that depth_image does not contain any zeros
    depth_image = np.maximum(depth_image, 1e-5)

    depth_image = scale_ratio / depth_image

    x, y, z = pixel_to_point(depth_image, camera_intrinsics)
    point_image = np.stack((x, y, z), axis=-1)
    
    return point_image.reshape(-1, 3), color_image.reshape(-1, 3) / 255.0