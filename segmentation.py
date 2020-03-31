import numpy as np
import cv2
from statistics import geometric_mean


def read_input(dicom_file_path, seg_file_path):
    with open(dicom_file_path) as f:
        array = np.array([[int(x) for x in line.split()] for line in f])

    with open(seg_file_path) as f:
        seg = np.array([[int(x) for x in line.split()] for line in f])

    return array, seg


def avg_segmentation_value(layer, segmentation):
    rows, cols = layer.shape

    values = []
    for i in range(rows):
        for j in range(cols):
            if segmentation[i, j]:
                values.append(max(int(layer[i, j]), 1))

    return geometric_mean(values)


def clamp_to_byte(pixel):
    if pixel < 0:
        pixel = 0
    elif pixel > 255:
        pixel = 255

    return int(pixel)


def extract_segmentation_as_image(layer, segmentation, border=20):
    rows, cols = layer.shape

    min_x = min_y = max(rows, cols)
    max_x = max_y = 0

    for i in range(rows):
        for j in range(cols):
            if segmentation[i, j]:
                min_x = min(min_x, i)
                max_x = max(max_x, i)

                min_y = min(min_y, j)
                max_y = max(max_y, j)

    border = max(max_x - min_x, max_y - min_y)/border

    min_x = int(max(min_x - border, 0))
    min_y = int(max(min_y - border, 0))
    max_x = int(min(max_x + border, rows))
    max_y = int(min(max_y + border, cols))

    width = max_x - min_x
    height = max_y - min_y

    minor_img = np.zeros((width, height, 1), np.uint8)
    offset = int(avg_segmentation_value(layer, segmentation) - 256/2)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            normalized_pixel = layer[i, j] - offset
            minor_img[i - min_x, j - min_y] = clamp_to_byte(normalized_pixel)

    return minor_img, min_x, min_y


def prepare_image(image, adjust_contrast=True, denoise=True, blur=True):
    if adjust_contrast:
        image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)

    if denoise:
        image = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)

    if blur:
        image = cv2.blur(image, (2, 2))

    return image


def get_contour(segmentation):
    width, height = segmentation.shape
    contour = np.zeros((width, height), np.uint8)

    for x in range(width):
        for y in range(height):
            if segmentation[x, y]:
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if segmentation[x+i, y+j] == 0:
                            contour[x, y] = 255
                            break

    return contour


def main():
    test_dir = 'tests/input1/'
    layer, seg = read_input(test_dir + 'in.in', test_dir + 'seg.in')

    avg = avg_segmentation_value(layer, seg)

    minor_img, min_x, min_y = extract_segmentation_as_image(layer, seg)

    cv2.imshow('extraction', minor_img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
