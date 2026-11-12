import cv2
import numpy as np

def generate_views(image, angles=[-10, 0, 10]):
    h, w, _ = image.shape
    views = []
    transforms = []

    for angle in angles:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        views.append(rotated)

        # Store inverse transform for alignment
        M_inv = cv2.invertAffineTransform(M)
        transforms.append(M_inv)

    return views, transforms
