# YOLO Object Detection 🏷️
We Use Pretrained YOLO Model to Detect Objects

## Requierments 📦

```bash
pip install -r requirements.txt
```

## Donwload and Convert pre-trained YOLO-v3 ⏬ 
```bash
# YOLO V3
- wget https://pjreddie.com/media/files/yolov3.weights -O models/yolov3.weights
- python convert_to_checkpoints.py --weights ./models/yolov3.weights --output ./checkpoints/yolov3.tf

# YOLO-Tiny V3
- wget https://pjreddie.com/media/files/yolov3-tiny.weights -O models/yolov3-tiny.weights
- python convert.py --weights ./models/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny True
```

## Video Frame Extractor CLI Options 🎞️

```bash
--frame     Frame Threshold     #default: 1800 (Every Minutes)
--src       Video File PATH     #default: 'sample.mp4'
```

You Need to Use This Command:

```bash
python frame.py --frame FRAME_TH --src VIDEO_FILE
```

## Live Stream Frame Extractor CLI Options 📺

```bash
--frame     Frame Threshold             #default: 1800 (Every Minutes)
--src       Video File PATH             #default: ''
--dir       Save Frames Folder Name     #default: '' | (Auto-Generate UUID4)
```

You Need to Use This Command:

```bash
python frame.py --frame FRAME_TH --src VIDEO_FILE
```

## Object Detection 📋

```bash
--weights           Path To .tf File                    #default: 'model/model.h5'
--classes           Path To Classes File                #default: './models/coco.names'
--tiny              Use Tiny Model Or Not?              #default: False
--num_classes       Number Of Classes In The Model      #default: 80
--size              Resize Images To                    #default: 416
--image             Path To Input Image                 #default: './data/sample.jpg'
--save              Save Or Not                         #default: False
```

Then You Just Need To Run This:

```bash
# Image
python main.py --image PATH_TO_IMAGE

# Video
python video.py --dir PATH_TO_FRAMES_DIR
```

## Features ✨

- [x] Detect Default COCO Classes 
- [x] CLI
- [x] Image Files
- [x] Video Files
- [x] Live Stream
- [ ] Support I18N Classes
- [ ] Telegram Bot
- [ ] Rest API
    - [ ] Image Support
    - [ ] Video Support
    - [ ] GIF Support