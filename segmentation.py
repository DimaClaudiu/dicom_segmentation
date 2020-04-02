"""Provides organ segmentation given an approximate contour This module
provides utilities for pre-preocessing dicom images, extracting and splitting
organs, and segmenting a mask for a certain organ."""

import numpy as np
import cv2
from statistics import geometric_mean


def avg_segmentation_value(layer, segmentation):
    """Returns the geometric mean of the segmented part of the layer.

    Arguments:
        layer {np 2d array} -- Dicom image of an organ
        segmentation {np 2d binary array} -- Approximate contour of an organ inside layer

    Returns:
        float -- The geometric mean of the segemnted organs values.
    """
    rows, cols = layer.shape

    values = []
    for i in range(rows):
        for j in range(cols):
            if segmentation[i, j]:
                values.append(max(int(layer[i, j]), 1))

    return geometric_mean(values)


def get_contour(segmentation):
    """Gets the countour of a binary mask.

    Arguments:
        segmentation {np 2d array} -- The mask of an object

    Returns:
        np 2d binary array -- The contour of the given mask.
    """
    width, height = segmentation.shape
    contour = np.zeros((width, height), np.uint8)

    for x in range(width):
        for y in range(height):
            if segmentation[x, y]:
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if segmentation[x+i, y+j] == 0:
                            contour[x, y] = 1
                            break

    return contour


def clamp_to_byte(pixel):
    """Clamps the input between 0-255.

    Arguments:
        pixel {float} -- The approximate value of an 8bit pixel

    Returns:
        int -- The clamped value of the pixel casted to int
    """
    if pixel < 0:
        pixel = 0
    elif pixel > 255:
        pixel = 255

    return int(pixel)


def extract_segmentation_as_image(layer, segmentation, border_percent=0.05):
    """Convert part of a given dicom image into and 8bit image by converting
    the spaces such that organ information isn't lost.

    Arguments:
        layer {Hounsfield array} -- A dicom layer with a segmented organ
        segmentation {np 2d binary array} -- A binary mask of the organ

    Keyword Arguments:
        border_percent {float} -- Add a percentage-based border to the segmentation (default: {0.05})

    Returns:
        [(np 2d array, int, int)] -- The segmented organ converted to 8bit color space, left-corner coordinates of the segmentation
    """
    rows, cols = layer.shape

    min_x = min_y = max(rows, cols)
    max_x = max_y = 0

    # Finding the edges of the mask
    for i in range(rows):
        for j in range(cols):
            if segmentation[i, j]:
                min_x = min(min_x, i)
                max_x = max(max_x, i)

                min_y = min(min_y, j)
                max_y = max(max_y, j)

    #  Part of the organ might be outside the given mask, so we'll use a buffer
    border_size = max(max_x - min_x, max_y - min_y) * float(border_percent)

    # Offseting the edges with the giiven bordersize
    min_x = int(max(min_x - border_size, 0))
    min_y = int(max(min_y - border_size, 0))
    max_x = int(min(max_x + border_size, rows))
    max_y = int(min(max_y + border_size, cols))

    width = max_x - min_x
    height = max_y - min_y

    # Creating the segmented image
    minor_img = np.zeros((width, height), np.uint8)
    # The average value of the organ will be in the middle of the 8bit colorspace to avoid data loss
    offset = int(avg_segmentation_value(layer, segmentation) - 256/2)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            normalized_pixel = layer[i, j] - offset
            minor_img[i - min_x, j - min_y] = clamp_to_byte(normalized_pixel)

    return minor_img, min_x, min_y


def prepare_image(image, adjust_contrast=True, denoise=True, blur=True):
    """Applies given filters to the image with default values appropriate for
    dicom images.

    Arguments:
        image {cv2 image} -- Image to be filtered

    Keyword Arguments:
        adjust_contrast {bool} -- Wether the contrast should be raised (default: {False})
        denoise {bool} -- Remove noise from the image (default: {True})
        blur {bool} -- Blur the image with a medianBlur (default: {True})

    Returns:
        [cv2 image] -- The filtered image
    """
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
    """Returns the mask of an accurate segmentation based on a approximate
    segmentation and an aggresive threshold.

    Arguments:
        image {cv2 image} -- The image where the search will be perfomed
        thresh {np 2d binary array} -- An aggresive threshold mask of the organ
        aprox_seg {np 2d binary array} -- An aproximate segmentation of the organ

    Keyword Arguments:
        debug {bool} -- Set True to see the full contours given by watershed (default: {False})

    Returns:
        np 2d binary array -- An accurate mask of the organ
    """
    width, height = image.shape

    # Part of mask we're "sure" is the organ
    sure_fg = np.zeros((width, height, 1), np.uint8)
    # Part of mask we're "sure" isn't the organ
    sure_bg = np.zeros((width, height, 1), np.uint8)

    for i in range(width):
        for j in range(height):
            coord = i, j
            sure_bg[coord] = 255
            # If it is part of an aggresive threshold,
            # and part of an approximate contour, it should be the organ
            if aprox_seg[coord] and thresh[i][j]:
                sure_fg[coord] = 255
            # If it isn't part of either, it should be ignored
            elif not aprox_seg[coord] and not thresh[i][j]:
                sure_bg[coord] = 0

    # Parts we'll have to find what they are
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Get the markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Wathershed only works with rgb images
    backtorgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Watershed returnes markers of a background, and different foregrounds
    markers = cv2.watershed(backtorgb, markers)

    # In our case:
    # 1 - Background
    # 2 - The organ we're interested in
    # 3... - Other organs or bones

    if debug:
        marked = backtorgb
        marked[markers == -1] = [0, 255, 0]
        cv2.imshow('markers', marked)
        cv2.waitKey()

    # Since we started the search fron the organ, the first
    # foreground region will be the organ of interest
    mask = np.zeros((width, height), np.uint8)
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            if markers[i, j] == 2:
                mask[i, j] = 1

    return mask


def smooth_edges(mask, ksize):
    """Smoothes the edges and closes unsual holes inside organs.

    Arguments:
        mask {np 2d binary array} -- A mask to be cleanned up
        ksize {int touple} -- Kernel size, higher values will result in more aggresive smoothing

    Returns:
        np 2d binary array -- A cleaned up version of the mask
    """
    kernel = np.ones(ksize, np.uint8)
    width, height = mask.shape

    # We might "hit the edges" after dilating, so get a overhead
    bordersize = int(0.2 * width)
    smoothed = cv2.copyMakeBorder(
        mask,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # Dilating and blurring covers edge artifaces
    smoothed = cv2.dilate(smoothed, kernel, iterations=1)
    smoothed = cv2.blur(smoothed, tuple([2*x for x in ksize]))

    # Thresholding gives integer values from the blurs approximates
    _, smoothed = cv2.threshold(
        smoothed, 128, 255, cv2.THRESH_BINARY)

    # First open the edges and then close all the artifacts and unsual holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    _, smoothed = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    smoothed = cv2.morphologyEx(
        smoothed, cv2.MORPH_CLOSE, kernel, iterations=2)

    return smoothed[bordersize:bordersize+width, bordersize:bordersize+height]


def extract_mask(dicom_array, segmentation_array, sensitivity=0.7, ksize=(4, 4), debug=False):
    """Default procedure for extracting a organ mask from a dicom array.

    Arguments:
        dicom_array {np 2d array} -- A dicom array in Hounsfield units
        segmentation_array {np 2d binary array} -- An approximate contour of the organ of interest

    Keyword Arguments:
        sensitivity {float} -- How aggresive the inital guess of the organ should be (default: {0.7})
        ksize {tuple} -- Controls how smooth the resulting mask will be (default: {(4,4)})
        debug {bool} -- Set True to see each step of the process (default: {False})

    Returns:
        np 2d binary array -- Returns the accurate segmetation of the organ
    """

    # First extract the organ and convert it to 8bit color space
    minor_img, min_x, min_y = extract_segmentation_as_image(
        dicom_array, segmentation_array, border_percent=0.05)

    if debug:
        cv2.imshow('Extracted', minor_img)
        cv2.waitKey()

    # We'll use only a part of the image since it makes the process faster and
    # less prone to false negatives
    min_width, min_height = minor_img.shape

    # Apply filters to the image for better thresholding and segmentation
    adjusted = prepare_image(minor_img, adjust_contrast=False)

    if debug:
        cv2.imshow('Adjusted', adjusted)
        cv2.waitKey()

    # Get the average value of the top-most values of the approximate segmentation
    adjusted_avg = avg_segmentation_value(
        minor_img, segmentation_array[min_x:min_x+min_width, min_y:min_y+min_height])

    # We'll use it to have a guess at the organ ourselves
    # This should be an aggresive guess,
    # since it will drive the watershed algorithm's starting points later
    epsilon = sensitivity * 100
    thresh = cv2.inRange(adjusted, adjusted_avg, adjusted_avg + epsilon)

    if debug:
        cv2.imshow('Thresh', thresh)
        cv2.waitKey()

    # Erode any little artifacts and enforce separation of organs
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)

    water_mask = get_mask_watershed(
        adjusted, thresh, segmentation_array[min_x:min_x+min_width, min_y:min_y+min_height], debug=debug)

    mask = smooth_edges(water_mask*255, ksize)

    # Scale the mask up to the original size
    major_mask = np.zeros(dicom_array.shape, np.uint8)
    major_mask[min_x:min_x+min_width, min_y:min_y+min_height] = mask/255

    return major_mask


def overlay_mask(img_path, mask):
    """Overlays the mask over an input image source for better visualization.

    Arguments:
        img_path {string} -- Path to an image representation of the dicom file
        mask {np 2d binary array} -- Mask of the organ to be overlayedzation.

    Arguments:
        img_path {string} -- Path to an image representation of the dicom file
        mask {np 2d binary array} -- Mask of the organ to be overlayed
    """
    dicom = cv2.imread(img_path)

    width, height = mask.shape

    for i in range(width):
        for j in range(height):
            if mask[i, j]:
                dicom[i, j] = [0, 60, 0]

    cv2.imshow('final', dicom)
    cv2.waitKey()


def write_mask(path, mask):
    """
    Writes the mask to the given path in a space separated, row-wise manner.
    """
    width, height = mask.shape
    with open(path, 'w') as f:
        for i in range(width):
            for j in range(height):
                f.write(str(mask[i, j]) + ' ')
            f.write('\n')
