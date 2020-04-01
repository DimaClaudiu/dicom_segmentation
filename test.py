"""Runs the segmentation module on test inputs."""
from segmentation import *


def read_input(dicom_file_path, seg_file_path):
    with open(dicom_file_path) as f:
        array = np.array([[int(x) for x in line.split()] for line in f])

    with open(seg_file_path) as f:
        seg = np.array([[int(x) for x in line.split()] for line in f])

    return array, seg

def main():
        test_dir = 'tests/input1/'
        layer, seg = read_input(test_dir + 'in.in', test_dir + 'seg.in')

        avg = avg_segmentation_value(layer, seg)

        minor_img, min_x, min_y = extract_segmentation_as_image(
            layer, seg, border_percent=0.05)

        min_width, min_height = minor_img.shape

        adjusted = prepare_image(minor_img, blur=True)

        adjusted_avg = avg_segmentation_value(
            minor_img, seg[min_x:min_x+min_width, min_y:min_y+min_height])

    
        epsilon = 70
        thresh = cv2.inRange(adjusted, adjusted_avg, adjusted_avg + epsilon)


        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=3)

        water = get_mask_watershed(
            adjusted, thresh, seg[min_x:min_x+min_width, min_y:min_y+min_height], debug=True)


        mask = smooth_edges(water*255, (4, 4))

        major_mask = np.zeros(layer.shape, np.uint8)
        major_mask[min_x:min_x+min_width, min_y:min_y+min_height] = mask

        show_mask(test_dir + 'dicom.png', major_mask)


    if __name__ == '__main__':
        main()