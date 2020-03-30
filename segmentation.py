from statistics import geometric_mean


def read_input(dicom_file_path, seg_file_path):
    with open(dicom_file_path) as f:
        array = [[int(x) for x in line.split()] for line in f]

    with open(seg_file_path) as f:
        seg = [[int(x) for x in line.split()] for line in f]

    return array, seg


def avg_segmentation_value(layer, segmentation):
    rows = min(len(layer), len(segmentation))
    cols = min(len(layer[0]), len(segmentation[0]))

    values = []
    for i in range(rows):
        for j in range(cols):
            if segmentation[i][j]:
                values.append(max(int(layer[i][j]), 1))

    print(values)
    return geometric_mean(values)


def main():
    test_dir = 'tests/input1/'
    layer, seg = read_input(test_dir + 'in.in', test_dir + 'seg.in')

    avg = avg_segmentation_value(layer, seg)

    print(avg)


if __name__ == '__main__':
    main()
