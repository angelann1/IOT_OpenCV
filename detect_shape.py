import os
import platform
import cv2
import numpy as np

WIDTH, HEIGHT = 800, 600

if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"


def polygon(center, radius, sides, rotation=-90):
    cx, cy = center
    angles = np.radians(
        rotation + np.arange(sides) * 360 / sides
    )
    points = np.stack(
        [
            cx + radius * np.cos(angles),
            cy + radius * np.sin(angles)
        ],
        axis=1
    )
    return points.astype(np.int32)


def main():

    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    cv2.circle(image, (150, 120), 70, (0, 120, 220), -1)
    cv2.fillPoly(image, [polygon((400, 140), 70, 5)], (0, 150, 150))
    cv2.fillPoly(image, [polygon((650, 130), 70, 6, 0)], (180, 0, 179))
    triangle = np.array([[150, 300],[220, 450],[80, 450]], dtype=np.int32)
    cv2.fillPoly(image, [triangle], (0, 0, 200))
    cv2.rectangle(image, (300, 300), (420, 420), (0, 180, 1), -1)
    cv2.rectangle(image, (520, 320), (750, 430), (200, 0, 1), -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    result = image.copy()
    result[thick_edges > 0] = (255, 0, 255)

    cv2.imshow("Shapes - Thick Green Edges", result)

    filename = "shapes_thick_green_edges.jpg"
    cv2.imwrite(filename, result)
    print(f"Saved image to '{filename}'")

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()