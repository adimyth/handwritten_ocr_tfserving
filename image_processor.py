import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_splits(path, plot=False):
    image_splits = []
    # Load image, grayscale, Gaussian blur, adaptive threshold
    image = cv2.imread(path, 0)
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30
    )

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Sorting contours along x-axis
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(c)
            # for cases where padding issue occurs - save them as is
            word_img = image[y : y + h, x : x + w]
            try:
                # padding to right & bottom
                word_img = np.pad(word_img, ((0, 250-h), (0, 600-w)),
                            'constant',constant_values=255)
            except Exception as e:
                traceback.print_exc()
            image_splits.append(word_img)

    if plot:
        for img in image_splits:
            plt.figure()
            plt.imshow(img)
            plt.show()
    return image_splits


if __name__ == "__main__":
    _ = get_splits("temp.png", plot=True)
