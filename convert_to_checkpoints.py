import argparse
import numpy as np
from scripts.yolo_v3.models import YoloV3, YoloV3Tiny
from scripts.yolo_v3.utils import load_darknet_weights
import tensorflow as tf

def parse_args():
    desc = "Convert YOLO Weights To Checkpoints"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--weights', type=str, default='./models/yolov3.weights', help='Path To Weights File')
    parser.add_argument('--output', type=str, default='./checkpoints/yolov3.tf', help='Path To Output')
    parser.add_argument('--tiny', type=bool, default=False, help='yolov3 or yolov3-tiny')
    parser.add_argument('--classes', type=int, default=80, help='Number Of Classes In The Model')

    return parser.parse_args()

def main(_argv):
    args = parse_args()
    if args is None:
        exit()
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.tiny:
        yolo = YoloV3Tiny(classes=args.classes)
    else:
        yolo = YoloV3(classes=args.classes)
    
    yolo.summary()
    print('Model Created')

    load_darknet_weights(yolo, args.weights, args.tiny)
    print('Weights Loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    print('Sanity Check Passed')

    yolo.save_weights(args.output)
    print('Weights Saved')


if __name__ == '__main__':
    main()
