import numpy as np 
from numpy import load 
from numpy import savez_compressed
import cv2
from numpy import asarray

data = load("C:/Users/nadee/Documents/Mountains/mountains_Diego.npz")

src , tar = data['arr_0'],data['arr_1']
blurred = list()
for el in tar:
    blurred_image_rgb = cv2.GaussianBlur(el, (3, 3), 0)
    blurred.append(blurred_image_rgb)


savez_compressed("C:/Users/nadee/Documents/Mountains/gen2_blur_mountains_Diego.npz", asarray(blurred), asarray(tar))
print("Saved dataset: ","gen2_blur_mountains_Diego.npz")
