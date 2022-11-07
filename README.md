# YoloV5 implemented with DeepSORT for tracking of salmon.

## What to do:
- Follow requirements for setup at https://github.com/nwojke/deep_sort and https://github.com/ultralytics/yolov5
- NOTE: Pytorch version > 2.0. is supported in this version, so no need for installing old pytorch 1.7. version.
- NOTE: The same goes for Tensorflow.
- Define your custom .yaml file in the data folder, which should point to your train, val, and test data.
- Then run the detect_deepsort.py file with your specific parameters.
    - As an example:
    ``` console 
    !python detect_deepsort.py --weights fish.pt --data fish.yaml --device 0 --img 640 --save-txt
    ````
    This runs a custom model named fish.pt, and uses the fish.yaml file pointing at train, val and test data.
    

## How to retrain YoloV5 model?
The YoloV5 has a great guide for this at https://github.com/ultralytics/yolov5
