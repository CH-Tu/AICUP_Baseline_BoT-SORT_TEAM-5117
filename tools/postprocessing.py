import argparse
import os
import numpy as np
from collections import OrderedDict, Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help='the path of tracking result directory',
                        metavar='str', required=True)
    parser.add_argument('-o', '--output_path', help='the path of output directory',
                        metavar='str', required=True)
    parser.add_argument('-p', '--percentage', help='the percentage of threshold',
                        metavar='int', type=int, default=25)
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    for txt_name in os.listdir(args.input_path):
        with open(os.path.join(args.input_path, txt_name)) as f:
            lines = f.readlines()

        frame_ids = [line.split(',')[0] for line in lines]
        frame_ids = list(OrderedDict.fromkeys(frame_ids))

        track_ids = [line.split(',')[1] for line in lines]
        counts = Counter(track_ids)
        counts_nonone = {key: value for key, value in dict(counts).items() if value != 1}
        counts_one = {key: value for key, value in dict(counts).items() if value == 1}
        count_threshold = int(np.percentile(list(counts_nonone.values()), args.percentage))

        bboxes = np.array([line.split(',')[2:6] for line in lines]).astype(float)
        bbox_centers = bboxes[:, :2] + bboxes[:, 2:]/2
        bbox_areas = bboxes[:, 2] * bboxes[:, 3]
        bbox_center_min_threshold = np.percentile(bbox_centers, args.percentage, axis=0)
        bbox_center_max_threshold = np.percentile(bbox_centers, 100-args.percentage, axis=0)
        bbox_area_threshold = np.percentile(bbox_areas, args.percentage)

        new_lines = list(lines)
        id_buffer = []
        for frame_id in frame_ids:
            count_one_lines = [line
                for line in lines
                    if line.split(',')[0] == frame_id and line.split(',')[1] in counts_one.keys()
            ]
            line_indices = [index
                for index, line in enumerate(lines)
                    if line.split(',')[0] == frame_id and line.split(',')[1] in counts_one.keys()
            ]

            if len(count_one_lines) != 0:
                if len(id_buffer) == 0:
                    for index, line in zip(line_indices, count_one_lines):
                        track_id = line.split(',')[1]
                        count = 0
                        bbox_center = bbox_centers[index]
                        bbox_area = bbox_areas[index]
                        id_buffer.append({
                            'track_id': track_id,
                            'count': count,
                            'bbox_center': bbox_center,
                            'bbox_area': bbox_area
                        })
                else:
                    dists = []
                    for index in line_indices:
                        bbox_center = bbox_centers[index]
                        dists_temp = np.array([])
                        for bbox_center_temp in [temp['bbox_center'] for temp in id_buffer]:
                            dist = ((bbox_center - bbox_center_temp) ** 2).sum() ** (1/2)
                            dists_temp = np.append(dists_temp, dist)
                        dists.append(dists_temp.min())
                    buffer_indices = sorted(range(len(dists)), key=lambda i: dists[i])

                    new_ids = []
                    for line_index, buffer_index in zip(line_indices, buffer_indices):
                        try:
                            new_line = lines[line_index]
                            parts = new_line.split(',')
                            parts[1] = id_buffer[buffer_index]['track_id']
                            new_lines[line_index] = ','.join(parts)
                            id_buffer[buffer_index]['count'] += 1
                            id_buffer[buffer_index]['bbox_center'] = bbox_centers[line_index]
                            id_buffer[buffer_index]['bbox_area'] = bbox_areas[line_index]
                        except IndexError:
                            new_line = lines[line_index]
                            parts = new_line.split(',')
                            track_id = parts[1]
                            count = 0
                            bbox_center = bbox_centers[line_index]
                            bbox_area = bbox_areas[line_index]
                            new_ids.append({
                                'track_id': track_id,
                                'count': count,
                                'bbox_center': bbox_center,
                                'bbox_area': bbox_area
                            })
                    id_buffer = id_buffer + new_ids

            if len(id_buffer) != 0:
                del_index = []
                for i, temp in enumerate(id_buffer):
                    if (temp['count'] > count_threshold or
                        any(temp['bbox_center'] < bbox_center_min_threshold) or
                        any(temp['bbox_center'] > bbox_center_max_threshold) or
                        temp['bbox_area'] < bbox_area_threshold):
                        del_index.append(i)
                id_buffer = [temp for i, temp in enumerate(id_buffer) if i not in del_index]

        with open(os.path.join(args.output_path, txt_name), 'w') as f:
            f.writelines(new_lines)

if __name__ == '__main__':
    main()
