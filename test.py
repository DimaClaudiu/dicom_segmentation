"""Runs the segmentation module on test inputs."""
from segmentation import *


def read_input(dicom_file_path, seg_file_path):
    with open(dicom_file_path) as f:
        array = np.array([[int(x) for x in line.split()] for line in f])

    with open(seg_file_path) as f:
        seg = np.array([[int(x) for x in line.split()] for line in f])

    return array, seg


def test_input(test_dir):
    layer, seg = read_input(test_dir + 'in.in', test_dir + 'seg.in')

    mask = extract_mask(layer, seg, sensitivity=0.7, ksize=(4, 4), debug=True)

    write_mask(test_dir + 'my_seg.out', mask)
    overlay_mask(test_dir + 'dicom.png', mask)


if __name__ == '__main__':
    for i in range(4):
        test_input('tests/input' + str(i+1) + '/')
