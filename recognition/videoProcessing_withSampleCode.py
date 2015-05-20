# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:06:54 2015

@author: Shelly
"""

import numpy as np
import cv2
#from matplotlib import pyplot as plt

RECORD = 0
path = "C:/Users/Shelly/Google Drive/HSI/Nao code/videos/"
video = "videoB30m"
suffix = "Converted"
ext = ".avi"

#kernel = np.ones((5,5),np.float32)/25

cap = cv2.VideoCapture(path+video+suffix+ext)
#if cap.isOpened()==False:
#    print "Opening video failed, trying again..."
#    cap.open(path+video)

# Define the codec and create VideoWriter object
if RECORD:
    fourcc = cv2.VideoWriter_fourcc(*'Xvid')
    out = cv2.VideoWriter(path+video+"Red"+ext,fourcc, 20.0, (640,480))

#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

#trainSetB = [] # convert later with np.array(trainSetB)

timesDetected = 0
ret, frame = cap.read()

while(ret):
#    ret, frame = cap.read()
#    while ret==False:
#        print "Frame not found, trying again..."
#        ret, frame = cap.read()
        
#    fgmask = fgbg.apply(frame)
#    if fgmask==False:
#        print "fgmask did not work..."
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # From 
    # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    # use "cv2.cvtColor(np.uint8([[[0,0,0 ]]]),cv2.COLOR_BGR2HSV)" to find
#    lower_blue = np.array([110,50,50])
#    upper_blue = np.array([130,255,255])
    lower_red = np.array([0,190,180])
    upper_red = np.array([50,255,255])
    lower_arm = np.array([0,55,110])
    upper_arm = np.array([50,135,230])
        

    # Threshold the HSV image to get only the right colors
    maskRed = cv2.inRange(hsv, lower_red, upper_red)
#    maskArm = cv2.inRange(hsv, lower_arm, upper_arm)
#    cv2.imshow('arm mask',maskArm)

    # Bitwise-AND mask and original image
#    mask = cv2.bitwise_and(maskRed,maskArm,mask=None)
#    cv2.imshow('combined mask',mask)
    
#    res = cv2.bitwise_and(frame,frame,mask=maskRed)
#    cv2.imshow('test image',res)
    
#    dst = cv2.filter2D(res,-1,kernel)
#    blur = cv2.GaussianBlur(res,(7,7),0)
#    blur = cv2.bilateralFilter(res,9,75,75) #Slow!
#    gray = cv2.bilateralFilter(maskRed, 11, 17, 17)
    
#    blur = cv2.medianBlur(res,7)
    
        
#    for y in range(maskRed[0].size):
#        for x in range(maskRed[:,0].size):
#            if maskRed[x,y]==255:
#                trainSetB.append((x,y))
    
    
    maskBlur = cv2.medianBlur(maskRed,5)
    edges = cv2.Canny(maskBlur, 30, 200)
    if RECORD:
        out.write(edges)
    
    # find contours in the edge image, keep only the largest ones
    (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#    cv2.drawContours(edges, np.array(cnts), -1, (0,255,0), 3)
    
    # loop over our contours
    flagCnt = None
    for c in cnts:
        # approximate the contour
        perim = cv2.arcLength(c, True)
        approxCnt = cv2.approxPolyDP(c, 0.02 * perim, True)

        # if the approximated contour has four points, then
        # we can assume that we have found the flag
        if len(approxCnt) == 4:
            flagCnt = approxCnt
            timesDetected+=1
#            cv2.drawContours(frame, [flagCnt], -1, (0, 255, 0), 3)
#            cv2.imshow('frame',frame)
            break
    
    
#    cv2.imshow('mask',maskRed)
#    cv2.imshow('maskBlur',maskBlur)
#    cv2.imshow('edges',edges)
    
#    cv2.imshow('res',res)
#    cv2.imshow('blur',blur)
#    cv2.imshow('frame',frame)

#    plt.subplot(121),plt.imshow(res),plt.title('Original')
#    plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
#    plt.xticks([]), plt.yticks([])
#    plt.show()
    
#    try:
#        cv2.imshow('frame',fgmask)
#    except Exception, e:
#        print "imshow did not work; error was: ", e
    
    
#    k = cv2.waitKey(50) & 0xff # 20 fps -> wait 1000 ms/20 frames=50?
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    ret, frame = cap.read()

print "Times Detected = ", timesDetected

cap.release()
if RECORD:
    out.release()
cv2.destroyAllWindows()

video = "videoD25m"
cap = cv2.VideoCapture(path+video+suffix+ext)
if RECORD:
    out = cv2.VideoWriter(path+video+"Red"+ext,fourcc, 20.0, (640,480))

#trainSetD = []

timesDetected = 0
ret, frame = cap.read()

while(ret):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,205])
    upper_red = np.array([190,220,255])
    maskRed = cv2.inRange(hsv, lower_red, upper_red)
#    res = cv2.bitwise_and(frame,frame,mask=maskRed)
#    blur = cv2.medianBlur(res,7)
    
#    if RECORD:out.write(blur)
    
    
    maskBlur = cv2.medianBlur(maskRed,7)
    edges = cv2.Canny(maskBlur, 30, 200)
    if RECORD:
        out.write(edges)
    
    # find contours in the edge image, keep only the largest ones
    (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    # loop over our contours
    flagCnt = None
    for c in cnts:
        # approximate the contour
        perim = cv2.arcLength(c, True)
        approxCnt = cv2.approxPolyDP(c, 0.02 * perim, True)

        # if the approximated contour has four points, then
        # we can assume that we have found the flag
        if len(approxCnt) == 4:
            flagCnt = approxCnt
            timesDetected+=1
            cv2.drawContours(frame, [flagCnt], -1, (0, 255, 0), 3)
            cv2.imshow('frame',frame)
            break
    
    
#    cv2.imshow('frame',frame)
#    cv2.imshow('res',res)
#    cv2.imshow('blur',blur)
    
#    for y in range(maskRed[0].size):
#        for x in range(maskRed[:,0].size):
#            if maskRed[x,y]==255:
#                trainSetD.append((x,y))
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    ret, frame = cap.read()


print "Times Detected = ", timesDetected

cap.release()
if RECORD:
    out.release()
cv2.destroyAllWindows()




## From 
## http://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
## define the list of boundaries
#boundaries = [
#	([17, 15, 100], [50, 56, 200]),
#	([86, 31, 4], [220, 88, 50]),
#	([25, 146, 190], [62, 174, 250]),
#	([103, 86, 65], [145, 133, 128])
#]
## loop over the boundaries
#for (lower, upper) in boundaries:
#	# create NumPy arrays from the boundaries
#	lower = np.array(lower, dtype = "uint8")
#	upper = np.array(upper, dtype = "uint8")
# 
#	# find the colors within the specified boundaries and apply
#	# the mask
#	mask = cv2.inRange(image, lower, upper)
#	output = cv2.bitwise_and(image, image, mask = mask)
# 
#	# show the images
#	cv2.imshow("images", np.hstack([image, output]))
#	cv2.waitKey(0)