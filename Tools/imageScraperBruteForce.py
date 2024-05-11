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


def wait_random_time():
    # Generate a random float between 0.01 and 0.5
    wait_time = random.uniform(0.01, 0.05)

    # Wait for the random amount of time
    time.sleep(wait_time)

# Call the function to wait for a random amount of time

api_key = "2decb6fe3a5c7762639cfb9802e7b4da"
api_secret = "c81ea6e8a8793c82"

flickr = flickrapi.FlickrAPI(api_key, api_secret)
flickr.authenticate_via_browser(perms="read")

flickr = flickrapi.FlickrAPI(api_key, api_secret, format="etree")

def GetImage(url: str) -> MatLike:

    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")

    return image

url_type = "url_z"
i = 0
scraped_urls = set()
pg = 0

while len(scraped_urls) < 500:
    results = flickr.photos.search(
        page=str(pg),
        tag_mode="all",
        tags="mountain, alps, snow",
        extras=url_type,
        per_page=100,
        sort="relevance",
        geo_context="0",
        content_types="0",
    )
    pg += 1

    for data in results[0]:
        title = data.get("title")
        url = data.get(url_type)
        scraped_urls.update()
        if url != None and url not in scraped_urls:
            if i % 10 == 0:
                print(f"Got {i+1} photos")
            scraped_urls.add(url)
            wait_random_time()
            image_raw = GetImage(url)
            # decode the image with color
            image = cv2.imdecode(image_raw, cv2.IMREAD_COLOR)
            if image.shape[0] < 256 or image.shape[1] < 256:
                pass
            image_scaled = resize_and_center_crop_image(image, 256, 256)
            processed = PostProcesser.ProcessToImages(image_scaled)
            # cv2.imshow("img", image)
            # cv2.imshow("img_p", processed)
            # print(processed.shape)
            # cv2.waitKey(0)

            cv2.imwrite(f"testDataUnfiltered/raw/Mountain-{str(i)}.jpg", image)
            cv2.imwrite(
                f"testDataUnfiltered/scaled/Mountain-{str(i)}.jpg",
                image_scaled,
            )
            cv2.imwrite(
                f"testDataUnfiltered/scaled_processed/Mountain_processed-{str(i)}.jpg",
                processed,
            )
            i += 1


with open("testDataUnfiltered/scraped_urls.txt", "w") as file:
    for string in scraped_urls:
        file.write(string + "\n")
