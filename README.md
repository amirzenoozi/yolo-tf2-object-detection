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
python main.py --image './data/sample.jpg'
```


## Features ✨

- [x] Detect Default COCO Classes 
- [x] CLI
- [ ] Support I18N Classes
- [ ] Telegram Bot
- [ ] Video Files
- [ ] Dataset
- [ ] Rest API
    - [ ] Image Support
    - [ ] Video Support
    - [ ] GIF Support