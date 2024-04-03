from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import os
import matplotlib as plt

def integratedDai():
    stepSize = 0.05
    f=open(str((Path(r'C:\Users\johns\Documents\Coding\drone\yolov5test.txt').resolve().absolute())), "a")
    f.write("yolov5n 4/2/2024 at time \n")
    fpsList=[]
    timeList=[]
    confList=[]
    newConfig=False
    #yolov3t
    #nnPath=str((Path(r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov3_coco_416x416.blob').resolve().absolute()))
    n=0
    #yolov8t            
    nnPath = str((Path(r'C:\Users\johns\Documents\Coding\drone\depthaimodelzoo\yolov5n_coco_416x416.blob').resolve().absolute()))
                
    labelMap = ["sports ball"]
                
    syncNN=True
                
    pipeline = dai.Pipeline()
                
    camRgb = pipeline.create(dai.node.ColorCamera)
                
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
                
    xoutRgb = pipeline.create(dai.node.XLinkOut)
                
    nnOut = pipeline.create(dai.node.XLinkOut)

    #stereo stuff
    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

    xoutDepth.setStreamName("depth")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # Properties
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    # Config
    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.6, 0.6)

    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    config.roi = dai.Rect(topLeft, bottomRight)

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
    stereo.depth.link(spatialLocationCalculator.inputDepth)

    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
    #spatend
                
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
        # Output queue will be used to get the depth frames from the outputs defined above
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

        color = (255, 255, 255)

        print("Use WASD keys to move ROI!")
        def frameNorm(frame,bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2]=frame.shape[1]
            return(np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        def displayFrame(name, frame):
            color = (255, 0, 0)
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                #print(bbox[0] + 10)
                #print("and ")
                #print(bbox[1] + 20)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                #print(int(detection.confidence*100))
                #f.write("NN FPS- ", 1, "time-", "detection confidence - ", int(detection.confidence*100))
                print("FPS: ", counter/(time.monotonic() - startTime),"detections", detections, "confidence: ",detection.confidence, "\n")
                confList.append(int(detection.confidence*100))
                fpsList.append(counter/(time.monotonic() - startTime))
                timeList.append(time.monotonic()-startTime)
                #f.write("FPS: ", counter/(time.monotonic() - startTime),"detections", detections, "confidence: ",detection.confidence, "\n")
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.imshow(name,frame)
        while True:
            inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

            depthFrame = inDepth.getFrame() # depthFrame values are in millimeters

            depth_downscaled = depthFrame[::4]
            if np.all(depth_downscaled == 0):
                min_depth = 0  # Set a default minimum depth value when all elements are zero
            else:
                min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            spatialData = spatialCalcQueue.get().getSpatialLocations()
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                depthMin = depthData.depthMin
                depthMax = depthData.depthMax

                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
                cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
            # Show the frame
            cv2.imshow("depth", depthFrameColor)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('w'):
                if topLeft.y - stepSize >= 0:
                    topLeft.y -= stepSize
                    bottomRight.y -= stepSize
                    newConfig = True
            elif key == ord('a'):
                if topLeft.x - stepSize >= 0:
                    topLeft.x -= stepSize
                    bottomRight.x -= stepSize
                    newConfig = True
            elif key == ord('s'):
                if bottomRight.y + stepSize <= 1:
                    topLeft.y += stepSize
                    bottomRight.y += stepSize
                    newConfig = True
            elif key == ord('d'):
                if bottomRight.x + stepSize <= 1:
                    topLeft.x += stepSize
                    bottomRight.x += stepSize
                    newConfig = True
            elif key == ord('1'):
                calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
                print('Switching calculation algorithm to MEAN!')
                newConfig = True
            elif key == ord('2'):
                calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
                print('Switching calculation algorithm to MIN!')
                newConfig = True
            elif key == ord('3'):
                calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MAX
                print('Switching calculation algorithm to MAX!')
                newConfig = True
            elif key == ord('4'):
                calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MODE
                print('Switching calculation algorithm to MODE!')
                newConfig = True
            elif key == ord('5'):
                calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                print('Switching calculation algorithm to MEDIAN!')
                newConfig = True

            if newConfig:
                config.roi = dai.Rect(topLeft, bottomRight)
                config.calculationAlgorithm = calculationAlgorithm
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)
                newConfig = False
            n=n+1
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

    #plt.scatter(x,y)
    #plt.show()
    #plt.savefig("output1.png")
    f.write(str(confList))
    f.write("\n")
    f.write("\n")
    f.write(str(fpsList))
    f.write("\n")
    f.write("\n")
    f.write(str(timeList))
    f.write("\n")

    f.close()
    print("^_^")

def main() -> None:
    integratedDai()

if __name__=='__main__':
    main()
        
