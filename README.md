# Strada_UCLA_ITS
Strada Labs + UCLA Institute of Transportations Studies


## Overview

NOTE: This is very much a work in progress and not all features have been implemented yet.

This repository contains code analizes video footage to detect parking space utilization, double parking and bike lane blocking. 

While all of the code needed to run this program is availale on this repository the object detection model is not. To get the model please contact alberto at stradalabs do com. There will be a dropbox link to the most recent model in the future.

## Software Dependencies
- Python3
- Tensorflow
- OpenCV

A full list of dependencies will be listed in a requirements.txt file in the future.

This projects uses Tensorflow for the initial object detection and benefits greatly from a strong GPU. The code was developed using a NVIDIA GTX 1080 and runs at 15 FPS. 

## How to run

###Specify video source

In the `sinlge.py` file on line 13. Replace the location of the video you wish to analyze with the local location of your video.

`cap = cv2.VideoCapture('YOUR-VIDEO-PATH')`

###Specify model location

Once the Tensorflow object detection model has been downloaded, replace the path with you own local path to the model.

`model_path = "YOUR-MODEL-PATH"`

Once the video and model paths have been specified run the following command:

`python3 single.py`


