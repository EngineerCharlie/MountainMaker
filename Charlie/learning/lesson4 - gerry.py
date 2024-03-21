import cv2
import numpy as np


# Takes 4 points on an image and deforms the image to match the points
def onClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < 4:
            src_points.append([x, y])
            cv2.circle(img_copy, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow("Img", img_copy)


img = cv2.imread("gerry.png", cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=0.8, fy=0.8)
img_copy = img.copy()

src_points = []

cv2.namedWindow("Img")
cv2.setMouseCallback("Img", onClick)


cv2.imshow("Img", img_copy)
cv2.waitKey(0)

# Calculates new image size
cols = int(
    (
        max(
            (
                (src_points[0][0] - src_points[3][0]) ** 2
                + (src_points[0][1] - src_points[3][1]) ** 2
            )
            ** 0.5,
            (
                (src_points[1][0] - src_points[2][0]) ** 2
                + (src_points[1][1] - src_points[2][1]) ** 2
            )
            ** 0.5,
        )
    )
)
rows = int(
    (
        max(
            (
                (src_points[0][0] - src_points[1][0]) ** 2
                + (src_points[0][1] - src_points[1][1]) ** 2
            )
            ** 0.5,
            (src_points[2][0] - src_points[3][0]) ** 2
            + (src_points[2][1] - src_points[3][1]) ** 2,
        )
        ** 0.5
    )
    / 2
)
ratio = rows / cols
cols = max(cols, 600)
rows = int(ratio * cols)

src_float = np.array(src_points, dtype=np.float32)

dest_points = np.array([[0, 0], [0, rows], [cols, rows], [cols, 0]], np.float32)
M = cv2.getPerspectiveTransform(src_float, dest_points)
out_img = cv2.warpPerspective(img, M, (cols, rows))
cv2.imshow("Img", out_img)
cv2.waitKey(0)
