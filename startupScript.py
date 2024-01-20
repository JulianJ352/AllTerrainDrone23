#!/usr/bin/env python3

#notes
# 1/20 first draft

#references:

# https://github.com/switchdoclabs/SDL_Pi_MJPEGStream/blob/master/streamtest.py

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import mavsdk
from glob import glob
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import io
import logging
import SocketServer
from threading import Condition
from PIL import ImageFont, ImageDraw, Image
import traceback
import StringIO
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from torch.cuda import amp



nnPath = str((Path(__file__).parent / Path('.../models/yolo...')).resolve().absolute())
if 1 < len(sys.argv):
    arg = sys.argv[1]
    if arg == "yolo3":
        nnPath = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    elif arg == "yolo4":
        nnPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    else:
        nnPath = arg
else:
    print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

labelMap = [ "sports ball"]

syncNN = True

pipeline = dai.Pipeline()

#sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xout.Rgb.setStreamName("rgb")
nnOut.setStreamName("nn")

#properties
camrgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorcameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFPS(40)

#network settings
detectionNetwork.setConfdienceThreshold(0.5)
detectionNetwork.setNumClasses(1)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchorMasks({"side26": [1,2,3], "side13":[3,4,5] })
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

#linking
camRbg.preview.link(detectionNetwork.input)
if syncNN:
  detectionNetwork.passthrough.link(xoutRgb.input)
else:
  camRgb.preview.link(xoutRgb.input)

#connect to device, start pipeline

with dai.Device(pipeline) as device:

  #output queues will get rgb frames
  qRgb = device.getOutputQueue(name="rgb", maxSize=4, )
  qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

  frame = None
  detections = []
  startTime = time.monotonic()
  counter = 0
  color2 = (255, 255, 255)

  def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0x, 1) * normVals).astype(int)

    def displayFrame(name, frame):
      color = (255, 0, 0)
      for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

      cv2.imshow(name,frame) #use as template to write frames for output?


while True:
  if syncNN:
    inRgb = qRgb.get()
    inDet = qDet.get()
  else:
    inRgb = qRgb.tryGet()
    inDet = qDet.tryGet()

  if in Rgb is not None:
    frame = inRgb.getCvFrame()
    cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

  if inDet is not None:
    detections = inDet.detections
    counter+=1

  if frame is not None:
    displayFrame("rgb", frame)

  if cv2.waitKey(1) = ord('q'):
    break


  #end video stream











#from SDL Pi MJPEG Stream's cvgrab.py
#link: https://github.com/switchdoclabs/SDL_Pi_MJPEGStream/blob/master/cvgrab.py

cap = cv2.VideoCapture(path)

ret, frame = cap.read()
print("found frame")

cv2.imwrite("test.jpg", frame)
print "done"
cap.release()
print("release done")
