
import cv2
from cv2.typing import MatLike, Point, Size
import numpy as np

class ColorConverter(object):

    colorList = [
        np.array([0, 255, 0]),  # light green for grass
        np.array([0, 128, 0]),  # dark green for forest
        np.array([139, 69, 19]),  # brown for mountain
        np.array([255, 255, 255]),  # white for snow
        np.array([255, 0, 0])  # blue for sky
    ]

    def rgb_to_lab(color):
        """
        Convert RGB color to CIE Lab color space.
        """

        # Convert RGB to XYZ
        rgb_to_xyz_matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                                    [0.2126729, 0.7151522, 0.0721750],
                                    [0.0193339, 0.1191920, 0.9503041]])
        xyz = np.dot(color, rgb_to_xyz_matrix.T)

        # Normalize XYZ
        xyz_ref = np.array([0.950456, 1.0, 1.088754])
        xyz_normalized = xyz / xyz_ref

        # Reshape xyz_normalized to be 2-dimensional
        xyz_normalized = np.reshape(xyz_normalized, (-1, 3))

        # Nonlinear transformation to Lab
        epsilon = 0.008856
        kappa = 903.3
        lab = np.zeros(3)
        lab[0] = np.where(xyz_normalized[:, 1] > epsilon, 116 * np.power(xyz_normalized[:, 1], 1/3) - 16, kappa * xyz_normalized[:, 1])
        lab[1] = 500 * (np.power(xyz_normalized[:, 0], 1/3) - np.power(xyz_normalized[:, 1], 1/3))
        lab[2] = 200 * (np.power(xyz_normalized[:, 1], 1/3) - np.power(xyz_normalized[:, 2], 1/3))

        return lab

    def color_distance(color1, color2):
        """
        Compute the Euclidean distance between two colors in Lab color space.
        """
        lab1 = ColorConverter.rgb_to_lab(color1)
        lab2 = ColorConverter.rgb_to_lab(color2)
        delta_l = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]
        distance = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        
        return distance
    
    skyblue = (255, 191, 0)
    
    def closest_color(input_color, color_list = colorList):
        """
        Find the color in the color_list that is closest to the input_color.
        """
        min_distance = float('inf')
        closest_color = None
        for color in color_list:
            distance = ColorConverter.color_distance(input_color, color)
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        
        return closest_color

    def closest_color2(color, color_list = colorList):
        color_list = np.array(color_list)
        color = np.array(color)
        distances = np.sqrt(np.sum((color_list-color)**2,axis=1))
        index_of_smallest = np.where(distances==np.amin(distances))
        smallest_distance = color_list[index_of_smallest]
        
        return smallest_distance 
