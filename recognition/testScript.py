# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:45:37 2015

@author: Shelly
"""
#
#import time, sys, select, os
#from naoqi import ALProxy

try:
   import cPickle as pickle
except:
   import pickle
ABSOLUTE=0
if ABSOLUTE:
    dataPath = "C:/Users/Shelly/Google Drive/HSI/Nao code/data/"
else:    
    import os
    dir = os.getcwd()
    dataPath = dir+"/data/"
dataExt = ".pickle"



files = ["videoB30m"]#,"videoD25m", "videoG25m", "videoM25m"]
dataSet = []
dataLen = []

for filename in files:
    print "Processing file for", filename
    dataFile = open(dataPath+filename+dataExt, 'rb')
    data = []
    
    while 1:
        try:
            data.append(pickle.load(dataFile))
        except EOFError:
            break
    
    dataFile.close()

print len(data)



#tts = ALProxy("ALTextToSpeech", "192.168.2.147", 9559)
##tts.say("Hello, world!")
#
##almemory = ALProxy("ALMemory", "192.168.2.147", 9559)
##pings = almemory.ping()
#
#motion = ALProxy("ALMotion", "192.168.2.147", 9559)
#motion.setStiffnesses("Body", 1.0)
#motion.moveInit()
#motion.post.moveTo(0.3, 0, 0)
#tts.say("I'm walking forward and talking")
#id = motion.post.moveTo(-0.3, 0, 0)
#motion.wait(id, 0)



#n = raw_input("Type 'stop' to stop recording\n")
#while n.strip() != "stop":
#    print "looping"
#    time.sleep(1)
#    n = raw_input("Type 'stop' to stop recording")


