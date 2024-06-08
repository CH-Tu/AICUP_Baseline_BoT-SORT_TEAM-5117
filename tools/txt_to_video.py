import argparse
import os
import random
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# from YOLOv7
def plot_one_box_PIL(bbox, image, color=None, label=None, line_thickness=None):
    draw = ImageDraw.Draw(image)
    line_thickness = line_thickness or max(int(min(image.size)/200), 2)
    draw.rectangle(bbox, width=line_thickness, outline=tuple(color))
    if label:
        fontsize = max(round(max(image.size)/40), 12)
        font = ImageFont.truetype('tools/Arial.ttf', fontsize)
        left, top, right, bottom = font.getbbox(label)
        txt_width = right - left
        txt_height = bottom - top
        draw.rectangle([bbox[0], bbox[1]-txt_height+4, bbox[0]+txt_width, bbox[1]], fill=tuple(color))
        draw.text((bbox[0], bbox[1]-txt_height+1), label, fill=(255, 255, 255), font=font)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', help='the path of image directory',
                        metavar='str', required=True)
    parser.add_argument('-t', '--txt_path', help='the path of tracking result txt directory',
                        metavar='str', required=True)
    parser.add_argument('-f', '--fps', help='the fps of output video',
                        metavar='int', type=int, default=2)
    args = parser.parse_args()

    random.seed(46)

    for file in os.listdir(args.txt_path):
        if os.path.splitext(file)[1] == '.txt':
            with open(os.path.join(args.txt_path, file)) as f:
                lines = f.readlines()

            frame_ids = [line.split(',')[0] for line in lines]
            track_ids = [line.split(',')[1] for line in lines]
            bboxes = np.array([line.split(',')[2:6] for line in lines]).astype(float) # [x0, y0, w, h]
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]                                # [x0, y0, x1, y1]

            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
            dir_name = os.path.splitext(file)[0]
            with imageio.get_writer(os.path.join(args.txt_path, f'{dir_name}.mp4'), fps=args.fps) as writer:
                for i, image_name in enumerate(sorted(os.listdir(os.path.join(args.image_path, dir_name)))):
                    image = Image.open(os.path.join(args.image_path, dir_name, image_name))

                    image_frame_id = str(i+1)
                    indices = [index for index, frame_id in enumerate(frame_ids) if frame_id == image_frame_id]
                    if len(indices) != 0:
                        for index in indices:
                            bbox = list(bboxes[index])
                            label = f'{track_ids[index]}, car'
                            color = colors[int(track_ids[index]) % len(colors)]
                            image = plot_one_box_PIL(bbox, image, color=color, label=label, line_thickness=2)

                    writer.append_data(np.array(image))

if __name__ == '__main__':
    main()
