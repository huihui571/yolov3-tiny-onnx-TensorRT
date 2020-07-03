import os
from onnx_to_tensorrt import YOLO

image_path = '/home/gantian/Experiments/yolov3-tiny-onnx-TensorRT/images/'
filelist = os.listdir(image_path)

yolo = YOLO()

for file in filelist:
    yolo.main(os.path.join(image_path, file))
