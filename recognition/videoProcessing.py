# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:06:54 2015

@author: Shelly
"""

import numpy as np
import cv2 ### NOTE: Needs OpenCV 3?
#from matplotlib import pyplot as plt
try:
   import cPickle as pickle
except:
   import pickle
from sklearn.cluster import MiniBatchKMeans


RECORD = 0
OUTPUT = 0
DISPLAY = 1

ABSOLUTE = 0
if ABSOLUTE:
    videoPath = "C:/Users/Shelly/Documents/GitHub/swarm_gesture_control/recognition/nao/videos/"
    dataPath = "C:/Users/Shelly/Documents/GitHub/swarm_gesture_control/recognition/data/"
else:
    import os
    dir = os.getcwd()
    videoPath = dir+"/nao/videos/"
    dataPath = dir+"/data/"
suffix = "Converted"
videoExt = ".avi"
dataExt = ".pickle"

XMAX=639
YMAX=479


# From http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged



def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh



def draw_detections(img, rects, color=(0, 255, 0), thickness = 2):
    for x,y,w,h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), color, thickness)


def checkCorners(rect):
    [x,y,w,h] = rect
    
    if x<0:
        w = w+x
        x=0
    elif x>XMAX:
        w = w-(x-XMAX)
        x=XMAX
        
    if y<0:
        h=h+y
        y=0
    elif y>YMAX:
        h = h-(y-YMAX)
        y=YMAX
        
    if (x+w)<0:
        w = x
    elif (x+w)>XMAX:
        w = XMAX-x
        
    if (y+h)<0:
        h=y
    elif (y+h)>YMAX:
        h = YMAX-y
        
    newRect = [x,y,w,h]
    return newRect


def normalizedRectArea(rect):
    newRect = checkCorners(rect)
    [x,y,w,h] = newRect
    
#    newRect = [x,y,w,h]
    return w*h


def trimRects(sorted_rects):
    new_rects = []
    
    for rect in sorted_rects:
        newRect = checkCorners(rect)
        [x,y,w,h] = newRect
        if w>0 and h>0:
            new_rects.append(newRect)
    
    return new_rects


def drawRect(rect, img):
    [x,y,w,h] = rect
    cv2.rectangle(img, (x,y), (x+w,y+h), 255, -1)
    return
    
    
def getROI(img,rect):
    [x,y,w,h] = rect
    return img[y:y+h,x:x+w]


def trimROI(rect):
    [x,y,w,h] = rect
    pad_w, pad_h = int(0.15*w), int(0.05*h)
    return ( x+pad_w, y, x+w-pad_w, y+h-pad_h )
#    return ( x+pad_w, y+pad_h, x+w-pad_w, y+h-pad_h )  # should technically be this



# Not finished - output not formatted correctly
#def rectToCnt(rect):
#    [x,y,w,h] = rect
#    return np.array( [(x,y),(x+w,y),(x,y+h),(x+w,y+h)] )


# From 
def colorQuant(image, clusters):
    (h, w) = image.shape[:2]
     
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
     
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((h*w, 3))
     
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
     
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
#    image = image.reshape((h, w, 3))
     
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
#    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
     
    return quant
    

def detectSkin(frame):
      # define the upper and lower boundaries of the HSV pixel
      # intensities to be considered 'skin'
    # 'regular' lighting
    lower1 = np.array([0, 70, 85], dtype = "uint8")
    upper1 = np.array([10, 165, 195], dtype = "uint8")
    # brighter
    lower2 = np.array([170, 50, 90], dtype = "uint8")
    upper2 = np.array([180, 120, 210], dtype = "uint8")
    lower4 = np.array([0, 50, 160], dtype = "uint8")
    upper4 = np.array([10, 80, 250], dtype = "uint8")
    # darker
    lower3 = np.array([0, 150, 60], dtype = "uint8")
    upper3 = np.array([15, 170, 90], dtype = "uint8")
    
      # resize the frame, convert it to the HSV color space,
    	# and determine the HSV pixel intensities that fall into
    	# the specifed upper and lower boundaries
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask1 = cv2.inRange(converted, lower1, upper1)
    skinMask2 = cv2.inRange(converted, lower2, upper2)
    skinMask3 = cv2.inRange(converted, lower3, upper3)
    skinMask4 = cv2.inRange(converted, lower4, upper4)
    skinMaskTemp = cv2.bitwise_or(skinMask1, skinMask2)
    skinMaskTemp = cv2.bitwise_or(skinMaskTemp, skinMask3)
    skinMask = cv2.bitwise_or(skinMaskTemp, skinMask4)
     
    	# apply a series of erosions and dilations to the mask
    	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
     
    	# blur the mask to help remove noise, then apply the
    	# mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3,3), 0)
#    cv2.imshow("skinMasks", np.hstack([skinMask3, skinMask]))
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    return skin


def whichArm(rect):
    ((x,y),(w,h),theta) = rect
    xbound = XMAX/2
#    ybound = YMAX/2
    if x<=xbound:
        return "left"
    elif x>xbound:
        return "right"



def processVideo(filename="videoB30m"):
    
    cap = cv2.VideoCapture(videoPath+filename+suffix+videoExt)
    if OUTPUT: dataFile = open(dataPath+filename+dataExt, 'wb')
    
    # Define the codec and create VideoWriter object
    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'Xvid')
        out = cv2.VideoWriter(videoPath+filename+"Human"+videoExt,fourcc, 20.0, (640,480))
    
#    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hogParams = {'winStride': (8,8), 'padding': (128,128), 'scale': 1.05}
    
    timesDetected = 0
    ret, frame = cap.read()
    
    while(ret):
        # From peopledetect.py
        # http://stackoverflow.com/questions/28476343/how-to-correctly-use-peopledetect-py
#        found,w = hog.detectMultiScale(frame, **hogParams)
#        found_filtered = []
#        for ri, r in enumerate(found):
#            for qi, q in enumerate(found):
#                if ri != qi and inside(r, q):
#                    break
#            else:
#                found_filtered.append(r)
##        print '%d (%d) found' % (len(found_filtered), len(found))
#                
#        
##        draw_detections(frame, found, (0, 0, 255))
###        draw_detections(frame, found_filtered, (0, 255, 0))
#        # Find largest box
#        found_sorted = sorted(found, key = normalizedRectArea, reverse = True)
#        
#        
#        if len(found_sorted)>0:
#            found_sorted = trimRects(found_sorted)  # NOT FINDING ALL CORRECT RECTS - fixed?
#        if len(found_sorted)>0:
#            largestFound = found_sorted[0]
##            largestCnt = rectToCnt(largestFound)
###            draw_detections(frame, [largestFound], (255, 0, 0))
#            
#            frameCrop = getROI(frame,largestFound)
#            frameCrop = cv2.fastNlMeansDenoisingColored(frameCrop,None,10,10,7,21)
#            cv2.imshow('frameCrop',frameCrop)
#            frameQuant = colorQuant(frameCrop,18)
        skin = detectSkin(frame)
#        if DISPLAY: cv2.imshow("frame, skin", np.hstack([frame, skin]))
        
        
        edges = cv2.Canny(skin,225,250)
        outline = np.zeros(skin.shape[:2], dtype = "uint8")
        (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#        if DISPLAY: cv2.drawContours(outline, cnts, -1, 255, -1)
            
        rects = []
        for idx,cnt in enumerate(cnts):
            # Upright bounding box
#                x,y,w,h = cv2.boundingRect(cnt)
#                cv2.rectangle(skin,(x,y),(x+w,y+h),(0,255,0),1)
#                print "upright aspect ratio:", (float(w)/h)
            # Rotated bounding box
            rect = cv2.minAreaRect(cnt)  # Returns a tuple: (x,y) of centroid, (w,h), and theta in deg bw horiz and first side (length?)
            box = np.int0( cv2.boxPoints(rect) )
            ((x,y),(w,h),theta) = rect
            if w<h: theta+=180
            else: theta+=90
            rect = [x,y,w,h,theta] # --- output format! ---
            if h!=0: 
                aspect_ratio = (float(w)/h)
                if aspect_ratio>=0.6 and aspect_ratio<=1.5:
                    if DISPLAY: cv2.drawContours(skin,[box],0,(0,0,255),1)
                else:
                    timesDetected+=1
#                        arm = whichArm(rect)
                    rects.append(rect)
                    if DISPLAY: cv2.drawContours(skin,[box],0,(0,255,0),1)
#                print "cnt", idx,"is rotated at", theta,"with aspect ratio:", aspect_ratio
#        if DISPLAY: cv2.imshow("edges, cnts", np.hstack([edges, outline]))
        if DISPLAY: cv2.imshow("frame, skin boxed", np.hstack([skin, frame]))
        if not DISPLAY: print "Times Detected:", timesDetected
            
#            if len(rects)>2:
#                rects=rects[:2]
#            elif len(rects)>1:
#                pass
        if OUTPUT and len(rects)>0: pickle.dump(rects, dataFile)

        
#        cv2.imshow('frame',frame)
#        if RECORD: out.write(frame)
        
        
#        k = cv2.waitKey(50) & 0xff # 20 fps -> wait 1000 ms/20 frames=50?
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        ret, frame = cap.read()
    
    
    print "Total Times Detected:", timesDetected
    
    if RECORD: out.release()
    
    cap.release()
    if OUTPUT: dataFile.close()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    # currently only using L to R videos
    filename = "videoA30m"
    print "Processing file ", filename 
    
    if 'B' in filename:  # status: done for one flag
        # define range of color in HSV
#        lower_red = np.array([0,190,180])
#        upper_red = np.array([50,255,255])
#        useHSV = 1
#        blurAmount = 5
#        numVertices = 4
        pass
    
        
    elif 'D' in filename:  # status: done for one flag
        # in HSV
#        lower_red = np.array([0,120,205])
#        upper_red = np.array([190,220,255])
#        useHSV = 1
#        blurAmount = 7
#        numVertices = 4
        pass
    
        
    elif 'G' in filename:  # status: good enough...
        # In BGR
#        lower_red = np.array([0,15,120])
#        upper_red = np.array([30,55,255])
#        useHSV = 0
#        blurAmount = 7
#        numVertices = 6
        pass
    
        
    elif 'M' in filename:  # status: IN PROGRESS
        # In BGR
#        lower_red = np.array([0,0,120])
#        upper_red = np.array([65,65,255])
#        useHSV = 0
#        blurAmount = 5
#        numVertices = 4
        pass
    
    
#    processVideo(filename, lower_red, upper_red, useHSV, blurAmount, numVertices)
    processVideo(filename)
    print "Done processing."
    if OUTPUT: print "See output data in", dataPath+filename+dataExt
    else: print "Data was not saved."
    





#def processVideo(filename="videoB30m"):
#    
#    cap = cv2.VideoCapture(videoPath+filename+suffix+videoExt)
#    if OUTPUT: dataFile = open(dataPath+filename+dataExt, 'wb')
#    
#    # Define the codec and create VideoWriter object
#    if RECORD:
#        fourcc = cv2.VideoWriter_fourcc(*'Xvid')
#        out = cv2.VideoWriter(videoPath+filename+"Human"+videoExt,fourcc, 20.0, (640,480))
#    
##    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
#    hog = cv2.HOGDescriptor()
#    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#    hogParams = {'winStride': (8,8), 'padding': (128,128), 'scale': 1.05}
#    
#    timesDetected = 0
#    ret, frame = cap.read()
#    
#    while(ret):
#        # From peopledetect.py
#        # http://stackoverflow.com/questions/28476343/how-to-correctly-use-peopledetect-py
#        found,w = hog.detectMultiScale(frame, **hogParams)
#        found_filtered = []
#        for ri, r in enumerate(found):
#            for qi, q in enumerate(found):
#                if ri != qi and inside(r, q):
#                    break
#            else:
#                found_filtered.append(r)
##        print '%d (%d) found' % (len(found_filtered), len(found))
#                
#        
##        draw_detections(frame, found, (0, 0, 255))
###        draw_detections(frame, found_filtered, (0, 255, 0))
#        # Find largest box
#        found_sorted = sorted(found, key = normalizedRectArea, reverse = True)
#        
#        
#        if len(found_sorted)>0:
#            found_sorted = trimRects(found_sorted)  # NOT FINDING ALL CORRECT RECTS - fixed?
#        if len(found_sorted)>0:
#            largestFound = found_sorted[0]
##            largestCnt = rectToCnt(largestFound)
###            draw_detections(frame, [largestFound], (255, 0, 0))
#            
##            cv2.imshow('frame',frame)
##            mask = np.zeros(frame.shape[:2],np.uint8)
##            drawRect(largestFound,mask)
##            mframe = cv2.bitwise_and(frame, frame, mask=mask)
##            cv2.imshow('masked frame',mframe)
##            if RECORD: out.write(frame)
#            
#            frameCrop = getROI(frame,largestFound)
#            frameCrop = cv2.fastNlMeansDenoisingColored(frameCrop,None,10,10,7,21)
##            cv2.imshow('frameCrop',frameCrop)
##            frameQuant = colorQuant(frameCrop,18)
#            skin = detectSkin(frameCrop)
#            if DISPLAY: cv2.imshow("frameCrop, (quant), skin", np.hstack([frameCrop, skin]))
#            
#            # Weighted avg bg subtraction
#            # From http://stackoverflow.com/questions/26344036/python-opencv-background-subtraction
##            avg1 = np.float32(frameQuant) # these should only be defined once...
##            avg2 = np.float32(frameQuant)
##            cv2.accumulateWeighted(frameQuant,avg1,1)
##            cv2.accumulateWeighted(frameQuant,avg2,0.01)
##            res1 = cv2.convertScaleAbs(avg1)
##            res2 = cv2.convertScaleAbs(avg2)
##            cv2.imshow("avg1, avg2", np.hstack([res1,res2]))
##            fgmask = fgbg.apply(frameQuant)
##            cv2.imshow("fgmask on frameQuant", fgmask)            
##            fgmask = fgbg.apply(frameCrop)
##            cv2.imshow('fgmask',fgmask)
#            
#        
##        # Use human bounding box for foreground extraction
##        # from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html
##            mask = np.zeros(frameCrop.shape[:2],np.uint8)
##    
##            bgdModel = np.zeros((1,65),np.float64)
##            fgdModel = np.zeros((1,65),np.float64)
##    
###            rect = (100,80,600,460)  # ROI rect input format is tuple!
##            cv2.grabCut(frameCrop,mask,trimROI( [0,0,frameCrop.shape[0],frameCrop.shape[1]] ),bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
##    
##            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
##            fgCrop = frameCrop*mask2[:,:,np.newaxis]
##            cv2.imshow('fgCrop',fgCrop)
#            
##            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#            
###            gray = cv2.cvtColor(frameCrop, cv2.COLOR_BGR2GRAY)
##            gray = cv2.equalizeHist(gray)
###            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
###            gray = clahe.apply(gray)
##            gray = cv2.medianBlur(gray,5)
##            gray = cv2.GaussianBlur(gray,kernel,0)
###            cv2.imshow('gray',gray)
##            ret1,th1 = cv2.threshold(gray,0,127,cv2.THRESH_BINARY_INV)
##            cv2.imshow('th1',th1)
##            th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
##            cv2.imshow('th2',th2)
##            threshAdapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)
##            cv2.imshow('threshAdapt',threshAdapt)
##            threshGauss = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
##            cv2.imshow('threshGauss',threshGauss)
##            ret2,threshOtsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##            cv2.imshow('threshOtsu',threshOtsu)
#            
#            
#            edges = cv2.Canny(skin,225,250)
#            outline = np.zeros(skin.shape[:2], dtype = "uint8")
#            (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
#            if DISPLAY: cv2.drawContours(outline, cnts, -1, 255, -1)
#                
#            rects = []
#            for idx,cnt in enumerate(cnts):
#                # Upright bounding box
##                x,y,w,h = cv2.boundingRect(cnt)
##                cv2.rectangle(skin,(x,y),(x+w,y+h),(0,255,0),1)
##                print "upright aspect ratio:", (float(w)/h)
#                # Rotated bounding box
#                rect = cv2.minAreaRect(cnt)  # Returns a tuple: (x,y) of centroid, (w,h), and theta in deg bw horiz and first side (length?)
#                box = np.int0( cv2.boxPoints(rect) )
#                if DISPLAY: cv2.drawContours(skin,[box],0,(0,0,255),1)
#                ((x,y),(w,h),theta) = rect
#                if w<h: theta+=180
#                else: theta+=90
#                rect = [x,y,w,h,theta]
#                if h!=0: 
#                    aspect_ratio = (float(w)/h)
#                    if aspect_ratio>=0.85 and aspect_ratio<=1.15: pass
#                    else:
#                        timesDetected+=1
##                        arm = whichArm(rect)
#                        rects.append(rect)
##                print "cnt", idx,"is rotated at", theta,"with aspect ratio:", aspect_ratio
#            if len(rects)>1:
#                pass
#            if len(rects)>2:
#                rects=rects[:2]
#            if OUTPUT and len(rects)>0: pickle.dump(rects, dataFile)
#                
#            if DISPLAY: cv2.imshow("edges, cnts", np.hstack([edges, outline]))
#            if DISPLAY: cv2.imshow("skin boxed", skin)
##            auto_edges = auto_canny(skin)
##            cv2.imshow('auto_edges',auto_edges)
#            
##            closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
##            cv2.imshow('closing',closing)
##            opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
##            cv2.imshow('opening',opening)
##            gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
##            cv2.imshow('gradient',gradient)
#            
#        
#
#        
#        # From 
#        # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
#        
##        if useHSV:
##            # Convert BGR to HSV
##            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
##            # Threshold the HSV image to get only the right colors
##            maskRed = cv2.inRange(hsv, lower_red, upper_red)
##        else:
##            maskRed = cv2.inRange(frame, lower_red, upper_red)
#    
#        # Bitwise-AND mask and original image to show only red parts
##        res = cv2.bitwise_and(frame,frame,mask=maskRed)    
##        blur = cv2.medianBlur(res,7)
#
#        
##        maskBlur = cv2.medianBlur(maskRed,blurAmount)
##        edges = cv2.Canny(maskBlur, 30, 200)
##        if RECORD: out.write(edges)
#        
#        
##        # find contours in the edge image, keep only the largest ones
##        (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
##        
##        # loop over our contours
##        flagCnt = None
##        for c in cnts:
##            # approximate the contour
##            perim = cv2.arcLength(c, True)
##            approxCnt = cv2.approxPolyDP(c, 0.02 * perim, True)
##            cv2.drawContours(frame, [approxCnt], -1, (255, 0, 0), 3)
###            timesDetected+=1
###            if OUTPUT: pickle.dump(approxCnt, dataFile)
##    
##            # if the approximated contour has four points, then
##            # we can assume that we have found the flag
##            if len(approxCnt) == numVertices:
##                flagCnt = approxCnt
##                timesDetected+=1
##                if OUTPUT: pickle.dump(flagCnt, dataFile)
##                cv2.drawContours(frame, [flagCnt], -1, (0, 255, 0), 3)
###                break
#        
#        
##        cv2.imshow('frame',frame)
##        if RECORD: out.write(frame)
#        
#        
##        k = cv2.waitKey(50) & 0xff # 20 fps -> wait 1000 ms/20 frames=50?
#        if cv2.waitKey(30) & 0xFF == ord('q'):
#            break
#        
#        ret, frame = cap.read()
#    
#    
#    print "Times Detected = ", timesDetected
#    
#    if RECORD: out.release()
#    
#    cap.release()
#    if OUTPUT: dataFile.close()
#    cv2.destroyAllWindows()



    
    
#def processVideo(filename="videoB30m", lower_red=np.array([0,190,180]), upper_red=np.array([50,255,255]), useHSV=1, blurAmount=5, numVertices=4):
#    
#    cap = cv2.VideoCapture(videoPath+filename+suffix+videoExt)
#    if OUTPUT: dataFile = open(dataPath+filename+dataExt, 'wb')
#    
#    # Define the codec and create VideoWriter object
#    if RECORD:
#        fourcc = cv2.VideoWriter_fourcc(*'Xvid')
#        out = cv2.VideoWriter(videoPath+filename+"Edges"+videoExt,fourcc, 20.0, (640,480))
#    
#    timesDetected = 0
#    ret, frame = cap.read()
#    
#    while(ret):        
#        # From 
#        # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
#        
#        if useHSV:
#            # Convert BGR to HSV
#            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#            # Threshold the HSV image to get only the right colors
#            maskRed = cv2.inRange(hsv, lower_red, upper_red)
#        else:
#            maskRed = cv2.inRange(frame, lower_red, upper_red)
#    
#        # Bitwise-AND mask and original image to show only red parts
##        res = cv2.bitwise_and(frame,frame,mask=maskRed)    
##        blur = cv2.medianBlur(res,7)
#
#        
#        maskBlur = cv2.medianBlur(maskRed,blurAmount)
#        edges = cv2.Canny(maskBlur, 30, 200)
#        if RECORD: out.write(edges)
#        
#        
#        # find contours in the edge image, keep only the largest ones
#        (_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#        
#        # loop over our contours
#        flagCnt = None
#        for c in cnts:
#            # approximate the contour
#            perim = cv2.arcLength(c, True)
#            approxCnt = cv2.approxPolyDP(c, 0.02 * perim, True)
#            cv2.drawContours(frame, [approxCnt], -1, (255, 0, 0), 3)
##            timesDetected+=1
##            if OUTPUT: pickle.dump(approxCnt, dataFile)
#    
#            # if the approximated contour has four points, then
#            # we can assume that we have found the flag
#            if len(approxCnt) == numVertices:
#                flagCnt = approxCnt
#                timesDetected+=1
#                if OUTPUT: pickle.dump(flagCnt, dataFile)
#                cv2.drawContours(frame, [flagCnt], -1, (0, 255, 0), 3)
##                break
#        
#        
##        cv2.imshow('mask',maskRed)
#        cv2.imshow('maskBlur',maskBlur)
##        cv2.imshow('edges',edges)
##        
##        cv2.imshow('res',res)
##        cv2.imshow('blur',blur)
#        cv2.imshow('frame',frame)
#        
#        
##        k = cv2.waitKey(50) & 0xff # 20 fps -> wait 1000 ms/20 frames=50?
#        if cv2.waitKey(30) & 0xFF == ord('q'):
#            break
#        
#        ret, frame = cap.read()
#    
#    
#    print "Times Detected = ", timesDetected
#    
#    if RECORD: out.release()
#    
#    cap.release()
#    if OUTPUT: dataFile.close()
#    cv2.destroyAllWindows()