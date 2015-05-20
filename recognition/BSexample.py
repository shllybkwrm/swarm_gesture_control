# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:06:54 2015

@author: Shelly
"""

import cv2

cap = cv2.VideoCapture('videoB30m.avi')
print cap

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

while(1):
    ret, frame = cap.read()
    if ~ret:
        print "Frame was not found"
        
    fgmask = fgbg.apply(frame)
    if fgmask==None:
        print "fgmask did not work"
    
    cv2.imshow('frame',fgmask)
#    try:
#        cv2.imshow('frame',fgmask)
#    except Exception, e:
#        print "imshow did not work; error was: ", e
    
    k = cv2.waitKey(50) & 0xff # 20 fps -> wait 1000 ms/20 frames=50?
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()