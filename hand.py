import cv2 as cv
import numpy as np

W, H = 512, 512
canvas = np.zeros((H, W, 3), np.uint8)  
colors = [
    (255, 0, 255),   # pink
    (0, 255, 255),   # yellow
    (255, 255, 0),   # cyan
    (0, 255, 0),     # green
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
]

x = 0
for i, color in enumerate(colors):
    cv.line(canvas, (x, 0), (x, H), color, 15)
    x += 90
cv.circle(canvas, (256, 256), 50, (200, 200, 200), -1)
cv.line(canvas, (206, 256), (306, 256), (0, 0, 0), 2)
cv.line(canvas, (256, 206), (256, 306), (0, 0, 0), 2)

cv.imshow("Disco", canvas)
cv.waitKey(0)
cv.destroyAllWindows()