import argparse
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', help='the path of output directory',
                        metavar='str', required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_path, 'txts'))

    for folder in os.listdir('runs/detect'):
        shutil.move(os.path.join('runs/detect', folder), args.output_path)

    txt_paths = []
    for root, dirs, files in os.walk(args.output_path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                txt_paths.append(os.path.join(root, file))
    for txt_path in txt_paths:
        shutil.copy(txt_path, os.path.join(args.output_path, 'txts'))

if __name__ == '__main__':
    main()
