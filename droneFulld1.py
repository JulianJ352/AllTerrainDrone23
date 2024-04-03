"""
	# P18: Project Monarch Drone script
	#Place this code on Pi and access via ssh:
	#ssh pi@luxonis.local -X
	#models available: yolov3, yolov4, yolov5n, yolov6t, yolov7tiny, yolov8n
	#640x352: yolov7tiny, yolov8n
	
	"""

import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import RPi.GPIO as GPIO
from DRV8825 import DRV8825

def select_file():
    filepath = filedialog.askopenfilename()
    print("Selected File:", filepath)

def claw_init():
    """
	# 1.8 degree: nema23, nema14
	# softward Control :
	# 'fullstep': A cycle = 200 steps
	# 'halfstep': A cycle = 200 * 2 steps
	# '1/4step': A cycle = 200 * 4 steps
	# '1/8step': A cycle = 200 * 8 steps
	# '1/16step': A cycle = 200 * 16 steps
	# '1/32step': A cycle = 200 * 32 steps
	"""
    try:
	Motor1 = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, mode_pins=(16, 17, 20))
	Motor2 = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, mode_pins=(21, 22, 27))

	except:
            Motor1.Stop()
            Motor2.Stop()
            print("motor fail")


    

def claw_forward():
    claw_init()
    Motor1.SetMicroStep('softward','fullstep')
    Motor1.TurnStep(Dir='forward', steps=200, stepdelay = 0.005)
    time.sleep(0.5)
    print("Claw Module: Forward")
    Motor1.Stop()
    

def claw_backward():
    claw_init()
    Motor1.TurnStep(Dir='backward', steps=200, stepdelay = 0.005)
    Motor1.Stop()
    print("Claw Module: Backward")


#yoloModelsDefaultpaths
def chooseModel(a)
    yolov3=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    yolov4t=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    yolov5n=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    yolov6t=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    yolov7t=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    yolov7640=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    yolov8n=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    yolov8n640=r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob'
    if a==3:
        return yolov3
    
    elif a=='4':
        return yolov4t
    elif a=='5':
        return yolov5n
    elif a=='6':
        return yolov6t
    elif a=='7':
        return yolov7t
    elif a=='8':
        return yolov8n

    

def droneObjDetection():
    nnPath = str((Path(r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob').resolve().absolute()))
            
    labelMap = ["sports ball"]
            
    syncNN=True
            
    pipeline = dai.Pipeline()
            
    camRgb = pipeline.create(dai.node.ColorCamera)
            
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
            
    xoutRgb = pipeline.create(dai.node.XLinkOut)
            
    nnOut = pipeline.create(dai.node.XLinkOut)
            
    xoutRgb.setStreamName("rgb")
            
    nnOut.setStreamName("nn")
            
    camRgb.setPreviewSize(416,416)
            
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            
    camRgb.setInterleaved(False)
            
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            
    camRgb.setFps(40)
            
    detectionNetwork.setConfidenceThreshold(0.2)
            
    detectionNetwork.setNumClasses(1)
            
    detectionNetwork.setCoordinateSize(4)
            
    detectionNetwork.setAnchors([10,14,23,27,37,58,81,82,135,169,344,319])
            
    detectionNetwork.setAnchorMasks({"side26": [1,2,3], "side13":[3,4,5]})
            
    detectionNetwork.setIouThreshold(0.5)
            
    detectionNetwork.setBlobPath(nnPath)
            

    detectionNetwork.setNumInferenceThreads(2)
            
    detectionNetwork.input.setBlocking(False)
            
    camRgb.preview.link(detectionNetwork.input)
            
    if syncNN:
            detectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    detectionNetwork.out.link(nnOut.input)
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        frame = None
        detections=[]
        startTime=time.monotonic()
        counter=0
        color2 = (255, 255, 255)
        def frameNorm(frame,bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2]=frame.shape[1]
            return(np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        def displayFrame(name, frame):
            color = (255, 0, 0)
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.imshow(name,frame)
        while True:
            if syncNN:
                inRgb = qRgb.get()
                inDet = qDet.get()
            else:
                inRgb = qRgb.tryGet()
                inDet = qDet.tryGet()
            if inRgb is not None:
                frame = inRgb.getCvFrame()
                cv2.putText(frame, "NN fps: {:.2f}".format(counter/(time.monotonic() - startTime)), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            if inDet is not None:
                detections = inDet.detections
                counter+=1
            if frame is not None:
                displayFrame("rgb", frame)
            if cv2.waitKey(1) == ord('q'):
                break




def proceed_to_next_screen():
    # Destroy the current screen
    title_screen.destroy()
    # Create the next screen
    create_next_screen()

def create_next_screen():
    next_screen = tk.Tk()
    next_screen.title("Next Screen")
    
    # Set screen size
    next_screen.geometry(f"{screen_width}x{screen_height}")
    # Configure dark theme
    next_screen.configure(bg="black")

    # Add widgets to the next screen
    label1 = tk.Label(next_screen, text="Object Detection", fg="white", bg="black")
    label1.place(relx=0.5, rely=0.2, anchor="center")

    label2 = tk.Label(next_screen, text="Claw Module", fg="white", bg="black")
    label2.place(relx=0.5, rely=0.7, anchor="center")

    select_button = tk.Button(next_screen, text="Select File", command=select_file)
    select_button.place(relx=0.5, rely=0.4, anchor="center")

    forward_button = tk.Button(next_screen, text="Forward", command=claw_forward)
    forward_button.place(relx=0.3, rely=0.8, anchor="center")

    backward_button = tk.Button(next_screen, text="Backward", command=claw_backward)
    backward_button.place(relx=0.7, rely=0.8, anchor="center")

    # Run the next screen's event loop
    next_screen.mainloop()

# Create the title screen
title_screen = tk.Tk()
title_screen.title("Title Screen")

# Set screen size
screen_width = 600
screen_height = 400
title_screen.geometry(f"{screen_width}x{screen_height}")

# Configure dark theme
title_screen.configure(bg="black")

# Add monarch butterfly image
butterfly_image = Image.open(r"C:\Users\johns\Downloads\monarch.png")
#butterfly_image = butterfly_image.resize((100, 100), Image.ANTIALIAS)
butterfly_photo = ImageTk.PhotoImage(butterfly_image)

butterfly_label = tk.Label(title_screen, image=butterfly_photo, bg="black")
butterfly_label.place(relx=0.5, rely=0.1, anchor="center")

# Add widgets to the title screen
title_label = tk.Label(title_screen, text="P18: Project Monarch", fg="white", bg="black", font=("Helvetica", 24))
title_label.place(relx=0.5, rely=0.3, anchor="center")

team_label = tk.Label(title_screen, text="Bailey Allen, Cameron Cage, Robert Dally, Julian Johnson, Jaliyah Hoyt, Danny Lu", fg="white", bg="black", font=("Helvetica", 24))
team_label.place(relx=0.5, rely=0.5, anchor="center")

advisors_label = tk.Label(title_screen, text="Dr. Nikitopoulos, Dr. Walker, Dr. Koppelman, Jack Hawkins", fg="white", bg="black", font=("Helvetica", 24))
advisors_label.place(relx=0.5, rely=0.8, anchor="center")

proceed_button = tk.Button(title_screen, text="Mission Start!", command=proceed_to_next_screen, bg="white", fg="black", font=("Helvetica", 14))
proceed_button.place(relx=0.5, rely=0.9, anchor="center")

# Run the title screen's event loop
title_screen.mainloop()
