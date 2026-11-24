import cv2 as CV
import numpy as np
import os
import platform

if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

img = CV.imread("square.jpg")
gray = CV.cvtColor(img, CV.COLOR_BGR2GRAY)
edges = CV.Canny(gray, 50, 150, apertureSize=3)

lines = CV.HoughLines(edges, 1, np.pi / 180, 200)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    CV.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

while True:
    CV.imshow ("HoughLines", img)
    if CV.waitKey(1) & 0xFF == ord("q"):
        break
CV.imwrite("houghlines_output.jpg", img)