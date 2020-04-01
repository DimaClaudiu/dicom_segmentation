import numpy as np
import cv2
from statistics import geometric_mean


def avg_segmentation_value(layer, segmentation):
    rows, cols = layer.shape

    values = []
    for i in range(rows):
        for j in range(cols):
            if segmentation[i, j]:
                values.append(max(int(layer[i, j]), 1))

    return geometric_mean(values)


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


def clamp_to_byte(pixel):
    if pixel < 0:
        pixel = 0
    elif pixel > 255:
        pixel = 255

    return int(pixel)


def extract_segmentation_as_image(layer, segmentation, border_percent=0.05):
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

    border_size = max(max_x - min_x, max_y - min_y) * float(border_percent)

    min_x = int(max(min_x - border_size, 0))
    min_y = int(max(min_y - border_size, 0))
    max_x = int(min(max_x + border_size, rows))
    max_y = int(min(max_y + border_size, cols))

    width = max_x - min_x
    height = max_y - min_y

    minor_img = np.zeros((width, height), np.uint8)
    offset = int(avg_segmentation_value(layer, segmentation) - 256/2)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            normalized_pixel = layer[i, j] - offset
            minor_img[i - min_x, j - min_y] = clamp_to_byte(normalized_pixel)

    return minor_img, min_x, min_y


def prepare_image(image, adjust_contrast=False, denoise=True, blur=True):
    if adjust_contrast:
        image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)

    if denoise:
        image = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)

    if blur:
        image = cv2.medianBlur(image, 5)

    if adjust_contrast:
        image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)

    return image


def get_mask_watershed(image, thresh, aprox_seg, debug=False):
    width, height = image.shape

    sure_bg = np.zeros((width, height, 1), np.uint8)
    sure_fg = np.zeros((width, height, 1), np.uint8)

    boxes = []
    for i in range(width):
        for j in range(height):
            coord = i, j
            sure_bg[coord] = 255
            if aprox_seg[coord] and thresh[i][j]:
                sure_fg[coord] = 255
                boxes.append(coord)
            elif not aprox_seg[coord] and not thresh[i][j]:
                sure_bg[coord] = 0

    unknown = cv2.subtract(sure_bg, sure_fg)

    seed = boxes[-1]
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    backtorgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(backtorgb, markers)

    if debug:
        marked = backtorgb
        marked[markers == -1] = [0, 255, 0]
        cv2.imshow('markers', marked)
        cv2.waitKey()

    mask = np.zeros((width, height), np.uint8)
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            if markers[i, j] == 2:
                mask[i, j] = 1

    return mask


def smooth_edges(image, ksize):
    kernel = np.ones(ksize, np.uint8)
    width, height = image.shape

    bordersize = int(0.2 * width)
    smoothed = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    smoothed = cv2.dilate(smoothed, kernel, iterations=1)

    smoothed = cv2.blur(smoothed, tuple([2*x for x in ksize]))

    ret, smoothed = cv2.threshold(
        smoothed, 128, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    ret, smoothed = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    smoothed = cv2.morphologyEx(
        smoothed, cv2.MORPH_CLOSE, kernel, iterations=2)

    return smoothed[bordersize:bordersize+width, bordersize:bordersize+height]


def extract_mask(dicom_array, segmentation_array, sensitivity=0.7, ksize=(4,4), debug=False):
    minor_img, min_x, min_y = extract_segmentation_as_image(
        dicom_array, segmentation_array, border_percent=0.05)

    min_width, min_height = minor_img.shape

    adjusted = prepare_image(minor_img, blur=True)

    adjusted_avg = avg_segmentation_value(
        minor_img, segmentation_array[min_x:min_x+min_width, min_y:min_y+min_height])


    epsilon = sensitivity * 100
    thresh = cv2.inRange(adjusted, adjusted_avg, adjusted_avg + epsilon)


    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)

    water = get_mask_watershed(
        adjusted, thresh, segmentation_array[min_x:min_x+min_width, min_y:min_y+min_height], debug=debug)


    mask = smooth_edges(water*255, ksize)

    major_mask = np.zeros(dicom_array.shape, np.uint8)
    major_mask[min_x:min_x+min_width, min_y:min_y+min_height] = mask
    
    return major_mask
    

def overlay_mask(img_path, mask):
        dicom = cv2.imread(img_path)

        width, height = mask.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j]:
                    dicom[i, j] = [0, 60, 0]

        cv2.imshow('final', dicom)
        cv2.waitKey()
