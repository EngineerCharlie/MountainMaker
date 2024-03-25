import cv2
import numpy as np
import ImageTools

for i in range(1, 5):
    filename = "Charlie/night-sky-scld-" + str(i) + ".jpg"
    img = cv2.imread(filename)

    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Light blur to clean the image
    # img_grey = cv2.medianBlur(img_grey, 5)
    # # extract contours to 8 bit with 5 wide kernel
    # edges = cv2.Laplacian(img_grey, cv2.CV_8U, ksize=5)
    # # clean edge image (keep only the strong edges) by applying a threshold
    # # THRESH_BINARY_INV means background black and contours white
    # ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    # color_img = cv2.bilateralFilter(img, 9, 75, 75)  # d= dimension of kernel
    # # COnvert grayscale image to colour (ie give it 3 channels)
    # skt = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    # # join images to make cartoon??
    # # filtered_img = cv2.addWeighted(color_img, 0.8, skt, 0.2, 0)
    canny = cv2.cvtColor(cv2.Canny(img, 50, 200, None, 3), cv2.COLOR_GRAY2BGR)
    # canny = ImageTools.flood_fill(canny, [0, 0], (255, 0, 0))
    # redCol = ImageTools.reduce_colors(img, 5)
    # # filtered_img = cv2.bitwise_and(redCol, canny)
    # # filtered_img = cv2.addWeighted(redCol, 1, canny, 0.5, 0)
    # cv2.resize(skt, None, fx=0.5, fy=0.5)
    cv2.imshow("ORIGINAL", img)
    # cv2.imshow("Image", skt)
    cv2.imshow("Cannyd", canny)
    # cv2.imshow("red cols", redCol)
    # cv2.imshow("Filtered", filtered_img)

    cv2.waitKey(0)
