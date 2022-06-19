from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from scripts.yolo_v3.models import (YoloV3)
from scripts.yolo_v3.dataset import transform_images

import numpy as np
import tensorflow as tf

import os
import time
import argparse

def parse_args():
    desc = "Frame Extractor"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--src', type=str, default='', help='What Is The Watch Directory Path?')

    return parser.parse_args()

class Watcher:
    def __init__(self, directory):
        self.observer = Observer()
        self.directory = os.path.abspath(directory)

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.directory, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print('Error')

        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        # elif event.event_type == 'modified':
        #     # Taken any action here when a file is modified.
        #     print(f'Received modified event - {event.src_path}.')

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print(f'Received created event - {event.src_path}.')
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            for physical_device in physical_devices:
                tf.config.experimental.set_memory_growth(physical_device, True)
            
            yolo = YoloV3(classes=80)
            yolo.load_weights('./checkpoints/yolov3.tf').expect_partial()
            class_names = [c.strip() for c in open('./models/coco.names').readlines()]
            img_raw = tf.image.decode_image(open(event.src_path, 'rb').read(), channels=3)

            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, 416)

            boxes, scores, classes, nums = yolo(img)
            for i in range(nums[0]):
                print('\t{}, {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i])))


if __name__ == '__main__':
    args = parse_args()
    if not args.src:
        exit()
    
    w = Watcher(args.src)
    w.run()