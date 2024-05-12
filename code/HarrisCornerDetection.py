import cv2
import numpy as np

def harris_corner_detection(image_path, alpha=0.04):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=alpha)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('Harris Corner Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

harris_corner_detection('cubes.png')
