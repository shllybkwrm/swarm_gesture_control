# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:06:54 2015

@author: Shelly
"""

import os
import numpy as np
import cv2 ### NOTE: Needs OpenCV 3!!
#from matplotlib import pyplot as plt
try:
   import cPickle as pickle
except:
   import pickle


RECORD = 0
OUTPUT = 0
ABSOLUTE = 0
if ABSOLUTE:
    videoPath = "C:/Users/Shelly/Google Drive/HSI/Nao code/videos/"
    dataPath = "C:/Users/Shelly/Google Drive/HSI/Nao code/data/"
else:
    dir = os.getcwd()
    videoPath = dir+"/videos/"
    dataPath = dir+"/data/"
suffix = "Converted"
videoExt = ".avi"
dataExt = ".pickle"




def processVideo(filename="videoB30m", lower_red=np.array([0,190,180]), upper_red=np.array([50,255,255]), useHSV=1, blurAmount=5, numVertices=4):
    
    cap = cv2.VideoCapture(videoPath+filename+suffix+videoExt)
    if OUTPUT: dataFile = open(dataPath+filename+dataExt, 'wb')
    
    # Define the codec and create VideoWriter object
    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'Xvid')
        out = cv2.VideoWriter(videoPath+filename+"Edges"+videoExt,fourcc, 20.0, (640,480))
    
    timesDetected = 0
    ret, frame = cap.read()
    
    while(ret):        
        # From 
        # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
        
        if useHSV:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Threshold the HSV image to get only the right colors
            maskRed = cv2.inRange(hsv, lower_red, upper_red)
        else:
            maskRed = cv2.inRange(frame, lower_red, upper_red)
    
        # Bitwise-AND mask and original image to show only red parts
#        res = cv2.bitwise_and(frame,frame,mask=maskRed)    
#        blur = cv2.medianBlur(res,7)

        
        maskBlur = cv2.medianBlur(maskRed,blurAmount)
        edges = cv2.Canny(maskBlur, 30, 200)
        if RECORD: out.write(edges)
        
        
        # find contours in the edge image, keep only the largest ones
        (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        
        # loop over our contours
        flagCnt = None
        for c in cnts:
            # approximate the contour
            perim = cv2.arcLength(c, True)
            approxCnt = cv2.approxPolyDP(c, 0.02 * perim, True)
            cv2.drawContours(frame, [approxCnt], -1, (255, 0, 0), 3)
#            timesDetected+=1
#            if OUTPUT: pickle.dump(approxCnt, dataFile)
    
            # if the approximated contour has four points, then
            # we can assume that we have found the flag
            if len(approxCnt) == numVertices:
                flagCnt = approxCnt
                timesDetected+=1
                if OUTPUT: pickle.dump(flagCnt, dataFile)
                cv2.drawContours(frame, [flagCnt], -1, (0, 255, 0), 3)
#                break
        
        
#        cv2.imshow('mask',maskRed)
        cv2.imshow('maskBlur',maskBlur)
#        cv2.imshow('edges',edges)
#        
#        cv2.imshow('res',res)
#        cv2.imshow('blur',blur)
        cv2.imshow('frame',frame)
        
        
#        k = cv2.waitKey(50) & 0xff # 20 fps -> wait 1000 ms/20 frames=50?
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        ret, frame = cap.read()
    
    
    print "Times Detected = ", timesDetected
    
    if RECORD: out.release()
    
    cap.release()
    if OUTPUT: dataFile.close()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    
    filename = "videoB30m"
#    filename = "videoD25m"
#    filename = "videoG25m"
#    filename = "videoM25m"  # NOTE: Poss not working for B/D, need to check
    print "Processing file ", filename
    
    if 'B' in filename:  # status: done for one flag
        # define range of color in HSV
        lower_red = np.array([0,190,180])
        upper_red = np.array([50,255,255])
        useHSV = 1
        blurAmount = 5
        numVertices = 4
        
        
    elif 'D' in filename:  # status: done for one flag
        # in HSV
        lower_red = np.array([0,120,205])
        upper_red = np.array([190,220,255])
        useHSV = 1
        blurAmount = 7
        numVertices = 4
        
        
    elif 'G' in filename:  # status: "good enough"
        # In BGR
        lower_red = np.array([0,15,120])
        upper_red = np.array([30,55,255])
        useHSV = 0
        blurAmount = 7
        numVertices = 6
        
        
    elif 'M' in filename:  # status: IN PROGRESS
        # In BGR
        lower_red = np.array([0,0,120])
        upper_red = np.array([65,65,255])
        useHSV = 0
        blurAmount = 5
        numVertices = 4
    
    
    processVideo(filename, lower_red, upper_red, useHSV, blurAmount, numVertices)
    
    