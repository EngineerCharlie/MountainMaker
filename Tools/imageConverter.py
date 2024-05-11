import cv2
import numpy as np
from urllib.request import urlopen
import cv2
from cv2.typing import MatLike
from ImageTools import resize_and_center_crop_image
from Processer import PostProcesser

import xml.etree.ElementTree as ET
import flickrapi
import random
import time
import os

raw_data_dir = "testDataUnfiltered/raw"
file_list = os.listdir(raw_data_dir)

i = 0
for filename in file_list:
    # Assuming 'url_type' is defined elsewhere
    filepath = os.path.join(raw_data_dir, filename)
    if i % 10 == 0:
        print(f"Converted {i+1} photos")
    # decode the image with color
    image = cv2.imread(filepath)
    if image.shape[0] < 256:
        pass
    image_scaled = resize_and_center_crop_image(image, 256, 256)
    processed = PostProcesser.ProcessToImages(image_scaled)
    cv2.imwrite(
        f"testDataUnfiltered/valid/Mountain-{str(i)}.jpg",
        image_scaled,
    )
    cv2.imwrite(
        f"testDataUnfiltered/drawing/Mountain_processed-{str(i)}.jpg",
        processed,
    )
    i += 1
