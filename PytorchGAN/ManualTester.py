
from PIL import Image
import requests
from urllib.request import urlopen

import xml.etree.ElementTree as ET
import flickrapi
import cv2
from cv2.typing import MatLike, Point, Size
import numpy as np

from Processer import PostProcesser
import random


def ShowWindow(image, title):
    cv2.imshow(f'{title}', image)
    cv2.moveWindow(f'{title}', 500, 500)




cv2.waitKey(0)
cv2.destroyAllWindows()