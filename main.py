from scripts.yolo_v3.models import (YoloV3, YoloV3Tiny)
from scripts.yolo_v3.dataset import transform_images
from scripts.yolo_v3.utils import draw_outputs

import numpy as np
import tensorflow as tf

import argparse
import time
import cv2
import os

def parse_args():
    desc = "Object Detection"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--classes', type=str, default='./models/coco.names', help='Path To Classes File')
    parser.add_argument('--weights', type=str, default='./checkpoints/yolov3.tf', help='Path To Weights File')
    parser.add_argument('--tiny', type=bool, default=False, help='YOLO v3 or YOLO-Tiny v3')
    parser.add_argument('--num_classes', type=int, default=80, help='Number Of Classes In The Model')
    parser.add_argument('--size', type=int, default=416, help='Resize Images To')
    parser.add_argument('--image', type=str, default='./data/sample.jpg', help='Path To Input Image')
    parser.add_argument('--save', type=bool, default=False, help='Save Output File ?')

    return parser.parse_args()



def main():
    args = parse_args()
    if args is None:
        exit()
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if args.tiny:
        yolo = YoloV3Tiny(classes=args.num_classes)
    else:
        yolo = YoloV3(classes=args.num_classes)

    yolo.load_weights(args.weights).expect_partial()
    class_names = [c.strip() for c in open(args.classes).readlines()]
    img_raw = tf.image.decode_image(open(args.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, args.size)

    start_time = time.time()
    boxes, scores, classes, nums = yolo(img)
    end_time = time.time()

    for i in range(nums[0]):
        print('\t{}, {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i])))

    if args.save:
        file_parts = os.path.abspath(args.image).split(".")
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        file_name = f'{file_parts[0]}_output.{file_parts[-1]}'
        cv2.imwrite(file_name, img)

if __name__ == '__main__':
    main()