from scripts.yolo_v3.models import (YoloV3, YoloV3Tiny)
from scripts.yolo_v3.dataset import transform_images
from scripts.yolo_v3.utils import draw_outputs

import tensorflow as tf
import numpy as np
import argparse
import warnings
import os
import time
import json

import scripts.utils

def parse_args():
    desc = "NSFW Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--classes', type=str, default='./models/coco.names', help='Path To Classes File')
    parser.add_argument('--weights', type=str, default='./checkpoints/yolov3.tf', help='Path To Weights File')
    parser.add_argument('--tiny', type=bool, default=False, help='YOLO v3 or YOLO-Tiny v3')
    parser.add_argument('--num_classes', type=int, default=80, help='Number Of Classes In The Model')
    parser.add_argument('--size', type=int, default=416, help='Resize Images To')
    parser.add_argument('--image', type=str, default='./data/sample.jpg', help='Path To Input Image')
    parser.add_argument('--save', type=bool, default=False, help='Save Output File ?')
    parser.add_argument('--dir', type=str, default='data/frames', help='What Is Images Directory?')

    return parser.parse_args()

def main():
    args = parse_args()
    if args is None:
        exit()


    warnings.filterwarnings("ignore")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if args.tiny:
        yolo = YoloV3Tiny(classes=args.num_classes)
    else:
        yolo = YoloV3(classes=args.num_classes)

    yolo.load_weights(args.weights).expect_partial()
    class_names = [c.strip() for c in open(args.classes).readlines()]
    frames_path = args.dir

    response  = {}
    frames = os.listdir(frames_path)
    folder_name = os.path.basename(frames_path)
    start_time = time.time()
    for item in frames:
            if os.path.isfile(f'{frames_path}/{item}'):
                img_raw = tf.image.decode_image(open(f'{frames_path}/{item}', 'rb').read(), channels=3)

                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, args.size)
                boxes, scores, classes, nums = yolo(img)

                item_key_name = item.split('.')
                response[item_key_name[0]] = {}
                for index in range(nums[0]):
                    response[item_key_name[0]][class_names[int(classes[0][index])]] = round((np.array(scores[0][index]) * 100), 2)

    end_time = time.time()
    print(f'Total Prediction Time: {end_time - start_time}')
    scripts.utils.write_json_file(response, f'{os.path.abspath(os.path.dirname(args.dir))}/{folder_name}_result.json')

if __name__ == '__main__':
    main()