

def read_input(dicom_file_path, seg_file_path):
    with open(dicom_file_path) as f:
        array = [[int(x) for x in line.split()] for line in f]

    with open(seg_file_path) as f:
        seg = [[int(x) for x in line.split()] for line in f]

    return array, seg


def main():
    test_dir = 'tests/input1/'
    array, seg = read_input(test_dir + 'in.in', test_dir + 'seg.in')


if __name__ == '__main__':
    main()
