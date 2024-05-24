#Load the necessary libraries
import skimage.io
import matplotlib.pyplot as plt
from skimage import exposure
import cv2

#Load and display the image
low_contrast = skimage.io.imread(fname='input//color-6.jpg')

#Apply Standard Equalization to the original image:
eq_image = exposure.equalize_hist(low_contrast)

#Apply Adaptive Histogram Equalization to the low contrast image: 
image_adapt = exposure.equalize_adapthist(low_contrast, clip_limit=0.03)

'''#Plot the original and the equalized images together for comparison.
f = plt.figure()
f.add_subplot(1,3, 1)
plt.imshow(low_contrast)
f.add_subplot(1,3, 2)
plt.imshow(eq_image)
f.add_subplot(1,3, 3)
plt.imshow(image_adapt)
plt.show(block=True)'''

# Apply image filters:
def image_filters(img):
        
    # Convert image to grayscale
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Adaptive Histogram Equalization
    adjusted_img = exposure.equalize_adapthist(img, clip_limit=0.03)

    return adjusted_img

plt.imshow(image_filters(low_contrast))
plt.show()