
import cv2
from cv2.typing import MatLike, Point, Size
import numpy as np
from urllib.request import urlopen
import random

import Converter

class PostProcesser(object):


    def __init__(self):
        print("self created successfully")
        
    def GetImageFromUrl(url:str) -> MatLike:
        
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        
        return image
    
    def GetImageFromPath(path:str) -> MatLike:
        
        #read the image with color
        image = cv2.imread(path, -1)

        #print(image.shape)
        #resize the image
        image = cv2.resize(image, (250, 250))

        return image

    def ApplyCartoonyEffect(img:MatLike) -> MatLike:
        #get grayscaled img
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #apply a light blur for cleaning the image a bit
        img_gray = cv2.medianBlur(img_gray, 5)

        #we use laplacian filter to extract the contours
        edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize = 5)

        #threshold the edges image to get only good edges
        #THRESH_BINARY_INV is to work with images with black background white contour
        #THRESH_BINARY is to work with images with white background black contour
        ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

        #use the bilateral filter with high values
        color_img = cv2.bilateralFilter(img, 10, 250, 250)


        #convert grayscale image back to color image
        skt = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        #lets do the bitwise and for merging sketch and color
        output = cv2.bitwise_and(color_img, skt)

        return output

    def ApplySketchEffect(image:MatLike, divisionScale:int) -> MatLike:

        im_gray = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE) # The image object
        im_gray = cv2.resize(im_gray, (250, 250))
       
        inverted = 255-im_gray
        blurred = cv2.GaussianBlur(inverted, (21,21),0)
        invertedblur = 255-blurred
        
        output = cv2.divide(im_gray, invertedblur, scale = divisionScale)
        
        #output = cv2.Laplacian(output, cv2.CV_8U, ksize = 1)

        #contour detection
        #canny = cv2.Canny(output, 30, 150)
        #output = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ret, output = cv2.threshold(output, 111, 255, cv2.THRESH_BINARY_INV)


        """
        #converting to binary
        #thresh = 127
        #im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        
        #get grayscaled img
        #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #apply a light blur for cleaning the image a bit
        #im_bw = cv2.medianBlur(im_bw, 5)

        #we use laplacian filter to extract the contours
        #edges = cv2.Laplacian(im_bw, cv2.CV_8U, ksize = 5)

        #threshold the edges image to get only good edges
        #THRESH_BINARY_INV is to work with images with black background white contour
        #THRESH_BINARY is to work with images with white background black contour
        ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

        #use the bilateral filter with high values
        output = cv2.bilateralFilter(im_bw, 10, 250, 250)
        """
        

        return output
    
    def GetContours(image:MatLike, gray:MatLike, useMorph:bool, threshold1:int, threshold2:int, _thickness = 1) -> MatLike:

        #cv2.addWeighted(im_gray, 10, dst = im_gray)
        #inverted = 255-im_gray
        #blurred = cv2.GaussianBlur(inverted, (17,17),0)
        #invertedblur = 255-blurred
        
        #output = cv2.divide(im_gray, invertedblur, scale = 256.0)
        
        #cv2.threshold(output, 50, 255, 1, output)
        
        #output = cv2.Laplacian(output, cv2.CV_8U, ksize = 1)

        #contour detection
        #edges = cv2.Canny(image, 35, 150)


        '''
        gray[0][0] = 255
        gray[0][249] = 255
        gray[249][0] = 255
        gray[249][249] = 255
        '''



        edges = cv2.Canny(gray, threshold1, threshold2)

        if(useMorph):
            #element = cv2.getStructuringElement(0, (1,1), (0, 0))
            #cv2.getStructuringElement(0, 3, 3)
            """
            kernel = np.array([[1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0],
                               [1, 0, 1, 0, 1],
                               [0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1]], np.uint8)
            """
            
            
            #kernel_cross = cv2.getStructuringElement(cv2.MORPH_OPEN, (3,3))
            #kernel_rect = np.ones((11,11), np.uint8)
            cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), edges)
            cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)), edges)

            #kernel_rect = np.ones((3,3), np.uint8)
            #cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_rect, edges)
            #cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel_rect, edges)
        
        #print(edges)



        #cv2.CONTOURS_MATCH_I3,
        (contours, b) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        
        '''
        # Create contours for the four sides of the image
        height, width = image.shape[:2]
        top_side_contour = np.array([[0, 0], [width, 0]])
        bottom_side_contour = np.array([[0, height], [width, height]])
        left_side_contour = np.array([[0, 0], [0, height]])
        right_side_contour = np.array([[width, 0], [width, height]])

        # Combine the contour lists
        contours = [top_side_contour, bottom_side_contour, left_side_contour, right_side_contour]

        # Extend the contour list with contours from Canny edges
        contours.extend(contoursResult)
        '''


        white = (255, 255, 255)
        purple = (240, 32, 160)

        #colored_image = image.copy()
        #colored_image = np.ones_like(image, dtype=np.uint8)
        white_image = np.full(image.shape, white)
        mean_colors = []
        
        # Create contours for the four edges of the image
        '''
        top_edge = np.array([[0, 0], [width, 0]])
        bottom_edge = np.array([[0, height], [width, height]])
        left_edge = np.array([[0, 0], [0, height]])
        right_edge = np.array([[width, 0], [width, height]])
        '''
        top_edge = ((0, 0), (250, 0))
        bottom_edge = ((0, 250), (250, 250))
        left_edge = ((0, 0), (0, 250))
        right_edge = ((250, 0), (250, 250))


        # Add the edge contours to the contour list
        #contours = contours + (top_edge)
        #contours = contours + (bottom_edge)
        #contours = contours + (left_edge)
        #contours = contours + (right_edge)

        #contours = contours + (0,0)
        #contours = contours + (0,249)
        #contours = contours + (249,0)
        #contours = contours + (249,249)

        #contours.append([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])

        #contours[0].append([[0, 0]])

        #print("num of contours: ", len(contours))

        '''
        minContourSize = 250
        final_contours = []
        for c in contours:
            if(cv2.contourArea(c) > minContourSize):
                final_contours.append(c)

        cv2.drawContours(white_image, final_contours, -1, color = (0,0,255), thickness=_thickness)
        '''
        cv2.drawContours(white_image, contours, -1, color = (0,0,255), thickness=_thickness)

        for c in contours:
            '''
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
            #mean_color = cv2.mean(image, mask=mask)[:3]
            #mean_color = Converter.ColorConverter.closest_color(mean_color)
            mean_color = (0, 255, 0)
            mean_colors.append(mean_color)

            mask = np.zeros(image.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
            mean_color_array = np.full(image.shape, mean_color[:3])
            #print(mean_color_array)
            # Assign mean_color_array to colored_image where the mask is true
            colored_image = np.where(mask == 255, mean_color_array, colored_image)
            '''
            
            # compute the center of the contour
            M = cv2.moments(c)
            
            if(M["m00"] != 0):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                #cv2.floodFill(colored_image, None, (cX, cY), (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)), 50, 100)
                #cv2.circle(colored_image, (cX, cY), 2, (255,255,255), -1)
            #cv2.fillPoly(colored_image, [contour], (0, 128, 0))
            #print(len(colored_image[colored_image == (0,128,0)]), "/ 187.500")

            #cv2.floodFill(colored_image, None, contour, (0, 128, 0), 0, 25)
            

        #indexes = np.argwhere(colored_image > (0,0,0))
        #print("INDEX 0 ", len(indexes), "shape: ", indexes.shape, " WHICH ARE: ", indexes)
        #while(len(indexes) > 1):
        final_image = np.full(image.shape, white)
        
        for x in range(100):
            # Get the indices of white pixels in the colored_image
            indexes = np.argwhere(np.all(white_image == [255, 255, 255], axis=-1))

            if len(indexes) > 0:
                print("INDEX  ", len(indexes), "shape: ", indexes.shape,  "WHICH ARE: ", indexes, " so first element is: ", (indexes[0][0], indexes[0][1]))

                toSample = random.randint(0, len(indexes)-1)
                seed_point = (indexes[toSample][1], indexes[toSample][0])  # Note the reversal of x and y coordinates
                fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                mask = np.zeros((252,252), np.uint8)
                #num, white_image, filled_mask, rect = cv2.floodFill(white_image, mask, seed_point, (100,100,100), loDiff=50, upDiff=100, flags = 4  | (255 << 8))
                num, white_image, filled_mask, rect = cv2.floodFill(white_image, mask, seed_point, (0,0,0), loDiff=50, upDiff=100)

                cv2.circle(white_image, seed_point, 4, (200,0,0), -1)
            
                #filled_mask = cv2.convertScaleAbs(filled_mask)
                #filled_mask = filled_mask.astype(cv2.CV_8U)
                print("ORIG SHAPE", white_image.shape, " MASK SHAPE: ", filled_mask.shape)
                
                print("mask ", filled_mask)
                
                #original_size_mask = cv2.resize(filled_mask, (250, 250), interpolation=cv2.INTER_NEAREST)
                original_size_mask = filled_mask[1:-1, 1:-1]

                #_, binary_mask = cv2.threshold(filled_mask, 128, 255, cv2.THRESH_BINARY)

                #print("ORIG SHAPE", colored_image.shape, " MASK SHAPE: ", filled_mask.shape, "  BI ", binary_mask.shape)

                #white_image = cv2.convertScaleAbs(white_image)
                #white_image = white_image.astype(np.uint8)

                filled_region = cv2.bitwise_and(image, image, mask=original_size_mask)
                mean_color = cv2.mean(filled_region, mask=original_size_mask)[:3]
                #cv2.floodFill(colored_image, None, seed_point, mean_color, 50, 100)
                
                #closest_color_to_mean = Converter.ColorConverter.closest_color2(mean_color)
                closest_color_to_mean = mean_color

                final_image[original_size_mask != 0] = closest_color_to_mean

                
            else:
                break

            #indexes = []
            # Find the coordinates of the filled pixels
            #filled_pixels = np.where(mask == 255)

            # Remove the filled pixels from the indexes list
            #indexes = list(zip(range(mask.shape[1]), range(mask.shape[0])))
            #indexes = [index for index in indexes if index not in zip(filled_pixels[1], filled_pixels[0])]

        

        #print(contours)

        #print(mean_colors)
       
        cv2.drawContours(final_image, contours, -1, color = (0,0,0), thickness=_thickness)
    
        '''
        for y in range(colored_image.shape[0]):
            for x in range(colored_image.shape[1]):

                if(tuple(colored_image[x][y]) == purple):
                    cv2.floodFill(colored_image, None, (x,y), (0, 128, 0), 0, 25)
        '''

        final_image = cv2.convertScaleAbs(final_image)
        final_image = final_image.astype(np.uint8)

        return final_image

    def GetEdges(image:MatLike) -> MatLike:
        
        im_gray = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE) # The image object
        im_gray = cv2.resize(im_gray, (250, 250))
       
        #enhance contrast
        cv2.equalizeHist(im_gray, im_gray)

        #median blur
        #im_gray_blurred = cv2.medianBlur(im_gray, 21)
        
        #gaussian blur
        im_gray_blurred = cv2.GaussianBlur(im_gray, (21,21), 0)
       

        #Canny
        #binary_edge_map = cv2.Canny(im_gray_blurred, 25, 150)

         # Apply Sobel edge detection in both x and y directions
        sobel_x = cv2.Sobel(im_gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(im_gray_blurred, cv2.CV_64F, 0, 1, ksize=3)
    
        # Compute the gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
        # Threshold the gradient magnitude to obtain binary edge map
        threshold = 50  # Adjust threshold as needed
        binary_edge_map = np.uint8(gradient_magnitude > threshold) * 255



        edges = cv2.cvtColor(binary_edge_map, cv2.COLOR_GRAY2BGR)

        return edges
   
    def GetColors(image:MatLike) -> MatLike:
    
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (250, 250))
        #color detect
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color thresholds (blue color example)
        #colors in BGR
        lower_blue = np.array([125, 0, 0])
        upper_blue = np.array([255, 180, 180])

        # Create a mask
        blueMask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Define the color you want to replace the masked areas with (here: red)
        replacement_Blue = (255, 200, 130)  # BGR format for OpenCV

        # Replace pixels in the original image that correspond to the masked areas with the replacement color
        image[blueMask != 0] = replacement_Blue

        # Apply the mask to the original image
        #result = cv2.bitwise_and(image, image, mask=mask)
        
        return image

    def GetGrayscaleResizedImg(image:MatLike) -> MatLike:

        #im_gray = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        #im_gray = cv2.resize(im_gray, (250, 250))
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return im_gray


    def GetNormalizedImg(image:MatLike) -> MatLike:

        norm_image = np.zeros((image.shape[0], image.shape[1]))
        image = cv2.normalize(image, norm_image, 0, 255, cv2.NORM_MINMAX)

        return image
    

    def GetHSVImg(image:MatLike) -> MatLike:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    def GetSkeletonImg(image:MatLike, kernelSize:int = 5, _iterations:int = 1) -> MatLike:

        kernel = np.ones((kernelSize, kernelSize),np.uint8)
        erosion = cv2.erode(image, kernel, iterations = _iterations)

        return erosion
    

    def GetDenoisedImg(image:MatLike) -> MatLike:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    

    def GetBlurredImg(image:MatLike, strength:int = 11) -> MatLike:
        return cv2.medianBlur(image, strength)
    

    def GetKMeanImg(image:MatLike, num_clusters) -> MatLike:
        
        # Reshape the image into a 1D array of pixel intensities
        pixels = image.reshape((-1, 1))

        # Convert to float32
        pixels = np.float32(pixels)

        # Define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back into uint8
        centers = np.uint8(centers)

        # Replace pixel values with cluster center values
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        return segmented_image
    
    def GetThresholded(image:MatLike) -> MatLike:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        return thresh

    def GetBinaryContoured(image:MatLike, gray:MatLike, thresh:MatLike) -> MatLike:

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask to draw the contours on
        contour_mask = np.zeros_like(gray)

        # Draw contours on the mask
        cv2.drawContours(contour_mask, contours, -1, (255, 0, 0), thickness=cv2.FILLED)

        # Find the mean color of each contour area
        mean_colors = []
        for contour in contours:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, (255, 0, 0), thickness=cv2.FILLED)
            mean_color = cv2.mean(image, mask=mask)[:3]  # Calculate mean color ignoring alpha channel
            mean_colors.append(mean_color)

        
        # Color the closed areas with the mean colors
        colored_image = image.copy()
        for contour, mean_color in zip(contours, mean_colors):
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (255, 0, 0), thickness=cv2.FILLED)
            colored_image = np.where(mask == 255, mean_color, colored_image)
        

        '''
        # Color the closed areas with the mean colors
        colored_image = image.copy().astype(np.float32)  # Convert to float32 for color calculations
        for contour, mean_color in zip(contours, mean_colors):
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            
            # Create an array of mean_color values with the same shape as colored_image
            mean_color_array = np.zeros_like(colored_image)
            mean_color_array[:, :, 0] = mean_color[0]
            mean_color_array[:, :, 1] = mean_color[1]
            mean_color_array[:, :, 2] = mean_color[2]

            colored_image = np.where(mask == 255, mean_color_array, colored_image)

        # Convert back to uint8 for display
        colored_image = np.clip(colored_image, 0, 255).astype(np.uint8)
        '''


        return contour_mask