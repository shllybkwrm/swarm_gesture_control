# -*- encoding: UTF-8 -*-
#
# This example demonstrates how to use the ALVideoRecorder module to record a
# video file on the robot.
#
# Usage: python vision_videorecord.py "robot_ip"
#

import sys
import time
from naoqi import ALProxy


def main(robotIP):
    
    # From alrobotposture.py
    try:
        postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    except Exception, e:
        print "Could not create proxy to ALRobotPosture; error was: ", e
    
    #postureProxy.goToPosture("Sit", 1.0)
    #postureProxy.goToPosture("SitRelax", 1.0)

    print "Robot posture is ", postureProxy.getPostureFamily()
    
    
    # From sensors_sonar.py
    # Connect to ALSonar module.
#    try:
#        sonarProxy = ALProxy("ALSonar", robotIP, PORT)
#    except Exception, e:
#        print "Could not create proxy to ALSonar; error was: ", e
#    
#    # Subscribe to sonars, this will launch sonars (at hardware level) and start data acquisition.
#    sonarProxy.subscribe("dataCollection") # defaults: period=30 ms, precision=1
#    
#    # Now you can retrieve sonar data from ALMemory.
#    try:
#        memoryProxy = ALProxy("ALMemory", robotIP, PORT)
#    except Exception, e:
#        print "Could not create proxy to ALMemory; error was: ", e
#    
#    # Get sonar l&r first echo (distance in meters to the first obstacle).
#    print "left sonar: ", memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value")
#    print "right sonar: ", memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value")
    
    
    # From sensors_getInertialValues.py
    # Get the Compute Torso Angle in radian
#    TorsoAngleX = memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
#    TorsoAngleY = memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
#    print ("Torso Angles [radian] X: %.3f, Y: %.3f" % (TorsoAngleX, TorsoAngleY))  
    
    
    try:
        videoRecorderProxy = ALProxy("ALVideoRecorder", robotIP, PORT)
    except Exception, e:
        print "Could not create proxy to ALVideoRecorder; error was: ", e
        
    # This records a 320*240 MJPG video at 10 fps. (CHANGED)
    # Note MJPG can't be recorded with a framerate lower than 3 fps.
    videoRecorderProxy.setResolution(2)  # 640x480; 3 = 1280*960
    videoRecorderProxy.setFrameRate(20)  # Max 30
    videoRecorderProxy.setVideoFormat("MJPG")
    videoRecorderProxy.startRecording("/home/nao/recordings/cameras", "videoN10m")#+str(time.time()) )
    
    
    #for i in xrange(0,10):
#    n = raw_input("Type 'stop' to stop recording\n")
#    while n.strip() != "stop":
#        time.sleep(1)
#        n = raw_input("Type 'stop' to stop recording")
    
    raw_input("Recording started. Press Enter when done...")
    
    # Video file is saved on the robot in the
    # /home/nao/recordings/cameras/ folder.
    videoInfo = videoRecorderProxy.stopRecording()
    
    print "Video was saved on the robot: ", videoInfo[1]
    print "Num frames: ", videoInfo[0]
    
#    sonarProxy.unsubscribe("dataCollection")
    
    
    
    
if __name__ == "__main__":
    IP = "192.168.2.147"
    PORT = 9559
    
    # Read IP address from first argument if any.
    if len(sys.argv) > 1:
        IP = sys.argv[1]
    else:
        print "Using default robot IP: ", IP
        print "To enter another IP use python recordVideoandSonar.py 'robot_ip'"
    
    main(IP)