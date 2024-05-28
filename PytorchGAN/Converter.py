
import cv2
from cv2.typing import MatLike, Point, Size
import numpy as np
import enum
import math

class CColor(enum.Enum):
    SKY = 0
    MOUNTAIN = 1
    FOREST = 2
    SNOW = 3
    CLOUD = 4
    GRASS = 5

class ColorConverter(object):

    #np.array([19, 69, 139]),  # brown for mountain
    #np.array([175, 175, 175]),  # light gray for clouds

    colorList = [
        np.array([255, 0, 0]),  # blue for sky
        np.array([30, 30, 30]),  # dark gray for mountain
        np.array([0, 128, 0]),  # dark green for forest
        np.array([255, 255, 255]),  # white for snow
        np.array([255, 255, 255]),  # light gray for clouds
        np.array([0, 255, 0]),  # light green for grass
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
    
    def GetClosestColor(input_color, color_list = colorList):
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

    def GetClosestColor2(color:tuple, centroid:tuple, YParam:tuple, color_list = colorList):
        
        #without considering centroid and YParams:
        '''
        color_list = np.array(color_list)
        color = np.array(color)
        distances = np.sqrt(np.sum((color_list-color)**2,axis=1))
        #index_of_smallest = np.where(distances==np.amin(distances))
        index_of_smallest = np.argmin(distances)
        smallest_distance = color_list[index_of_smallest]
        '''

        color_list = np.array(color_list) #shape 4x3
        color = np.array(color) #shape 3x1 I guess

        #distances is a list of values in the order: SKY, MOUNTAIN, FOREST, SNOW, CLOUD
        distances = np.sqrt(np.sum((color_list-color)**2,axis=1))

        #print(distances)
        #print(YParam)

        #in terms of logic
        #YParam[0] is the highest position of the region
        #YParam[1] is the lowest position of the region
        #but the highest position is a smaller number because we count the position from the top left corner

        #normalize the centroid Y value-> the top is 0.5 and the bottom is 1

        # we multiply by the normalized centroid for sky identification: the distance to sky is smaller the close you are to the top
        
        '''
        mapped_centroid_SKY = 0.3 + math.sqrt(centroid[1] / 220) * 0.7
        mapped_centroid_CLOUD = 0.5 + math.sqrt(centroid[1] / 220) * 0.5
        mapped_centroid_SNOW = 2.5 - math.sqrt(centroid[1] / 230) * 1


        distances[CColor.SKY.value] = distances[CColor.SKY.value] * mapped_centroid_SKY

        distances[CColor.CLOUD.value] = distances[CColor.CLOUD.value] * mapped_centroid_CLOUD

        distances[CColor.SNOW.value] = distances[CColor.SNOW.value] * mapped_centroid_SNOW
        '''

        if(YParam[0] < 5):
            distances[CColor.SKY.value] = distances[CColor.SKY.value]/2.5       #if the region is connected to the top IT IS VERY LIKELY TO BE SKY
            distances[CColor.CLOUD.value] = distances[CColor.CLOUD.value]/1.25     #and IT IS VERY LIKELY TO BE CLOUD
            distances[CColor.MOUNTAIN.value] = distances[CColor.MOUNTAIN.value]*2
        elif(centroid[1] < 25):
            distances[CColor.SKY.value] = distances[CColor.SKY.value]/2        #if the region is connected to the top IT IS LIKELY TO BE SKY
            distances[CColor.CLOUD.value] = distances[CColor.CLOUD.value]/1.1     #and IT IS LIKELY TO BE CLOUD
        else:
            distances[CColor.SKY.value] = distances[CColor.SKY.value]*2        #if the region is connected to the top IT IS UNLIKELY TO BE SKY
            distances[CColor.CLOUD.value] = distances[CColor.CLOUD.value]*2     #and IT IS UNLIKELY TO BE CLOUD

        if (YParam[1] > 245):
            distances[CColor.SKY.value] = 255 #if the region is connected to the bottom it CANNOT BE SKY
            distances[CColor.CLOUD.value] = 255 #it cannot be cloud

        if(centroid[1] > 100):
            distances[CColor.CLOUD.value] = 255 # if the region is in above pos 200 it CANNOT BE CLOUD


        index_of_smallest = np.argmin(distances)

        '''
        if(YParam[1] == 249):
            index_of_smallest = 4 #if the region is connected to the top we choose to color it SKY no matter what
        else:
            index_of_smallest = np.argmin(distances)
        '''

        smallest_distance = color_list[index_of_smallest]   
        return smallest_distance 
