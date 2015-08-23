# swarm_gesture_control

Main files:
recognition/dataProcessing.py
Reads data files received from videoProcessing.py, trains classifiers, and outputs results & figures.

recognition/videoProcessing.py
Reads input data files received from recordVideo.py and uses frame-by-frame arm detection to create output data files.

recognition/nao/recordVideo.py
Used on the Nao robot to record 640x480, 20fps video.
