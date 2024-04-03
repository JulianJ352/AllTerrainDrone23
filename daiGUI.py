import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import os
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import matplotlib as plt
#import RPi.GPIO as GPIO

dStoppa=True
#start stepper code
'''
MotorDir = [
    'forward',
    'backward',
]

ControlMode = [
    'hardward',
    'softward',
]

class DRV8825():
    def __init__(self, dir_pin, step_pin, enable_pin, mode_pins):
        self.dir_pin = dir_pin
        self.step_pin = step_pin        
        self.enable_pin = enable_pin
        self.mode_pins = mode_pins
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.setup(self.step_pin, GPIO.OUT)
        GPIO.setup(self.enable_pin, GPIO.OUT)
        GPIO.setup(self.mode_pins, GPIO.OUT)
        
    def digital_write(self, pin, value):
        GPIO.output(pin, value)
        
    def Stop(self):
        self.digital_write(self.enable_pin, 0)
    
    def SetMicroStep(self, mode, stepformat):
        """
        (1) mode
            'hardward' :    Use the switch on the module to control the microstep
            'software' :    Use software to control microstep pin levels
                Need to put the All switch to 0
        (2) stepformat
            ('fullstep', 'halfstep', '1/4step', '1/8step', '1/16step', '1/32step')
        """
        microstep = {'fullstep': (0, 0, 0),
                     'halfstep': (1, 0, 0),
                     '1/4step': (0, 1, 0),
                     '1/8step': (1, 1, 0),
                     '1/16step': (0, 0, 1),
                     '1/32step': (1, 0, 1)}

        print ("Control mode:",mode)
        if (mode == ControlMode[1]):
            print ("set pins")
            self.digital_write(self.mode_pins, microstep[stepformat])
        
    def TurnStep(self, Dir, steps, stepdelay=0.005):
        if (Dir == MotorDir[0]):
            print ("forward")
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 0)
        elif (Dir == MotorDir[1]):
            print ("backward")
            self.digital_write(self.enable_pin, 1)
            self.digital_write(self.dir_pin, 1)
        else:
            print ("the dir must be : 'forward' or 'backward'")
            self.digital_write(self.enable_pin, 0)
            return

        if (steps == 0):
            return
            
        print ("turn step:",steps)
        for i in range(steps):
            self.digital_write(self.step_pin, True)
            time.sleep(stepdelay)
            self.digital_write(self.step_pin, False)
            time.sleep(stepdelay)
'''
#end stepper code

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
        while dStoppa==True:
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

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super(App,self).__init__()

        # configure window
        self.title("P18-Monarch.py")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

         # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Oak-D AI Config", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,text="Start", command=self.sidebar_button_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="CloseProg", command=self.sidebar_button_event2)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        #self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="testing", command=self.sidebar_button_event)
        #self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        #self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        #self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        #self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
         #                                                              command=self.change_appearance_mode_event)
        #self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        #self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        #self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        #self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               #command=self.change_scaling_event)
        #self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))
        

        #sidebar 2
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=3, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Claw Control", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Grab", command=self.grab)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Release", command=self.release)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Full Forward!", command=self.stepperFullFwd)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        #self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Forward", command=self.sidebar_button_event)
        #self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        #self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Backwards", command=self.sidebar_button_event)
        #self.sidebar_button_5.grid(row=5, column=0, padx=20, pady=10)
        #self.sidebar_button_6 = customtkinter.CTkButton(self.sidebar_frame, text="Third!", command=self.sidebar_button_event)
        #self.sidebar_button_6.grid(row=6, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create textbox
        #self.textbox = customtkinter.CTkTextbox(self, width=250)
        #self.textbox.grid(row=4, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")

        # Add monarch butterfly image
        #butterfly_image = Image.open(r"C:\Users\johns\Documents\Coding\drone\finalProgram\monarchp18.png")
        #butterfly_image = butterfly_image.resize((100, 100), Image.ANTIALIAS)
        #butterfly_photo = ImageTk.PhotoImage(butterfly_image)

        #butterfly_label = tkinter.Label(title_screen, image=butterfly_photo, bg="black")
        #butterfly_label.place(relx=0.5, rely=0.1, anchor="center")
        #image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_images")
        #self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        #self.home_frame.grid_columnconfigure(0, weight=1)
        #self.large_test_image = customtkinter.CTkImage(Image.open(r"C:\Users\johns\Documents\Coding\drone\finalProgram\monarchp18.png"), size=(500, 150))
        #self.home_frame_large_image_label = customtkinter.CTkLabel(App, text="Project18: Monarch", image=self.large_test_image)
        #self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)



       # my_image = customtkinter.CTkImage(light_image=Image.open(r"C:\Users\johns\Documents\Coding\drone\finalProgram\monarchp18.png"),
                                          #dark_image=Image.open(r"C:\Users\johns\Documents\Coding\drone\finalProgram\monarchp18.png"),
                                          #size=(30, 30))

        #image_label = customtkinter.CTkLabel(App, image=my_image, text="")  # display image with a CTkLabel
        my_image = customtkinter.CTkImage(light_image=Image.open(r"C:\Users\johns\Documents\Coding\drone\finalProgram\monarchp18.png"), dark_image=Image.open(r"C:\Users\johns\Documents\Coding\drone\finalProgram\monarchp18.png"), size=(500, 500))
        image_label = customtkinter.CTkLabel(self, image=my_image, text="")  # display image with a CTkLabel
        image_label.place(relx=0.5,rely=0.3, anchor=customtkinter.CENTER)
        
        #self.textbox = customtkinter.CTkTextbox(master=self, width=2, corner_radius=0)
        #self.textbox.grid(row=3, column=1, sticky="nsew")
        #self.textbox.insert("0.0", "P18: Monarch\n Bailey Allen, Cameron Cage, Robert Dally, Jaliyah Hoyt, Julian Johnson, Danny Lu \n Dr. Walker, Dr. Koppelman, Dr. Nik \n")
        self.lowCenterFrame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.lowCenterFrame.grid(row=3, column=1, rowspan=2, sticky="nsew")
        #self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.lowCenterFrame, text="P18: Monarch\n Team: Bailey Allen, Cameron Cage, Robert Dally, \n Jaliyah Hoyt, Julian Johnson, Danny Lu \nAdvisors and Sponsors: Dr. Walker, Dr. Koppelman, Dr. Nikitopoulos \n Mr. Jack Hawkins \n", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.logo_label.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)


        

        

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        integratedDai()

    def startOakAI(self):
        integratedDai()

    def sidebar_button_event2(self):
        dStoppa=False

    def stopAll(self):
        dStoppa=False

    def stepperHalfFwd(self):
       print("halfFWD")
       
    def stepperFullFwd(self):
        print("full fwd")

    def grab(self):
        print("grabbing")

    def stepperHalfRev(self):
        print("halfrev")

    def stepperFullRev(self):
        print("fullrev")

    def release(self):
        print("release")

        


if __name__ == "__main__":
    app = App()
    app.mainloop()
