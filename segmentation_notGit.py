import numpy as np
import cv2
from statistics import geometric_mean


def read_input(dicom_file_path, seg_file_path):
    with open(dicom_file_path) as f:
        array = np.array([[int(x) for x in line.split()] for line in f])

    with open(seg_file_path) as f:
        seg = np.array([[int(x) for x in line.split()] for line in f])

    return array, seg


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

    minor_img = np.zeros((width, height), np.uint8)
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


def get_segmentation_watershed(image, thresh, seg, adjusted_img=None, start_x=0, start_y=0):
    width, height = image.shape

    if adjusted_img is None:
        adjusted_img = image

    width2, height2 = thresh.shape

    sure_bg = np.zeros((width, height, 1), np.uint8)
    sure_fg = np.zeros((width, height, 1), np.uint8)

    contour = get_contour(seg)
    boxes = []
    for i in range(start_x, start_x + width2):
        for j in range(start_y, start_y + height2):
            coord = i, j
            sure_bg[coord] = 255
            if seg[coord] and thresh[i-start_x][j - start_y]:
                sure_fg[coord] = 255
                boxes.append(coord)
            elif not seg[coord] and not thresh[i-start_x][j - start_y]:
                sure_bg[coord] = 0

    # cv2.imshow('fg', sure_fg)
    # cv2.waitKey()
    # cv2.imshow('bg', sure_bg)
    # cv2.waitKey()

    unknown = cv2.subtract(sure_bg, sure_fg)

    seed = boxes[-1]
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    backtorgb = cv2.cvtColor(adjusted_img, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(backtorgb, markers)

    mask = np.zeros((width, height), np.uint8)
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            if markers[i, j] == 2:
                mask[i, j] = 255

    return mask


def smooth_edges(image, strength=2):
    kernel = np.ones((4, 4), np.uint8)

    smoothed = image
    smoothed = cv2.dilate(smoothed, kernel, iterations=1)

    # smoothed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # for _ in range(strength):
    smoothed = cv2.blur(smoothed, (6, 6))
    #     smoothed = cv2.erode(smoothed, kernel, iterations=1)
    #     smoothed = cv2.dilate(smoothed, kernel, iterations=2)

    #     # smoothed = cv2.dilate(smoothed, kernel, iterations=2)

    ret, smoothed = cv2.threshold(
        smoothed, 128, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    (thresh, binRed) = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=2)
    smoothed = cv2.morphologyEx(
        smoothed, cv2.MORPH_CLOSE, kernel, iterations=2)

    return smoothed


def show_segmentation(image_path, mask):
    img = cv2.imread(image_path)
    width, height = mask.shape

    cv2.imshow('original', img)
    cv2.waitKey()

    for i in range(width):
        for j in range(height):
            if(mask[i, j]):
                img[i, j][1] += 60

    cv2.imshow('final', img)
    cv2.waitKey()


def main(test_dir):
    layer, seg = read_input(test_dir + 'in.in', test_dir + 'seg.in')

    avg = avg_segmentation_value(layer, seg)
    print(avg)

    minor_img, min_x, min_y = extract_segmentation_as_image(
        layer, seg, border=60)

    min_width, min_height = minor_img.shape

    adjusted = prepare_image(minor_img, blur=False)

    contour = get_contour(seg)

    adj_major_img = np.zeros(layer.shape, np.uint8)
    adj_major_img[min_x:min_x+min_width, min_y:min_y+min_height] = adjusted

    # cv2.imshow('adjusted', adj_major_img)
    # cv2.waitKey()

    adjusted_avg = avg_segmentation_value(
        minor_img, seg[min_x:min_x+min_width, min_y:min_y+min_height])

    print(adjusted_avg)

    epsilon = 0
    ret, thresh = cv2.threshold(
        adjusted, adjusted_avg-epsilon, adjusted_avg+epsilon, cv2.THRESH_BINARY)

    cv2.imshow('thresh', thresh)
    cv2.waitKey()

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)

    water = get_segmentation_watershed(
        layer, thresh, seg, adjusted_img=adj_major_img, start_x=min_x, start_y=min_y)

    mask = smooth_edges(water)
    cv2.imwrite(test_dir + 'my_seg.png', mask)

    show_segmentation(test_dir + 'dicom.png', mask)


if __name__ == '__main__':
    for i in range(3, 4):
        main('tests/input' + str(i+1) + '/')
