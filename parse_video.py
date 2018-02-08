# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 23:49:39 2017

@author: pasca
"""
from __future__ import print_function
import os
import imageio
import re
import numpy as np
from tqdm import tqdm


def ParseVideos(videoPath, imagePath, sampleRate, transformFun=None, 
                overwrite=False, shift=0):
    #shift is in seconds
    
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    
    if not overwrite:
        oldVideoIds = [f.split('_')[0] for f in os.listdir(imagePath) if f.endswith('.jpg')]
    
    for file in tqdm(os.listdir(videoPath)):
        filename = os.path.join(videoPath, file)
        print(filename)
        videoId = filename.split('.')[0]
        if not overwrite:
            if videoId in oldVideoIds:
                print('skip')
                continue
        
        try:
            reader = imageio.get_reader(filename)
        except OSError:
            with open("errors.txt", "a") as myfile:
                myfile.write(videoId+'\n')
            continue
            
        fps = reader.get_meta_data()['fps']
        duration = reader.get_meta_data()['duration']
        nFrame = reader.get_meta_data()['nframes']
        
        if sampleRate is not None:
            #calculate the time points in ms to sample frames
            timePoints = np.arange(shift*1000, duration*1000, 1000.0/sampleRate)
            timePoints = np.floor(timePoints).astype(int)
            #calculate the frame indexes
            frameIndexes = (np.floor(timePoints/1000.0*fps)).astype(int)
            nSample = len(frameIndexes)
        else:
            frameIndexes = np.arange(nFrame)
            timePoints = (frameIndexes*1000/fps).astype(int)
            nSample = nFrame
        
        for i in range(nSample):
            #read image
            image = reader.get_data(frameIndexes[i])
            #apply transformation
            if transformFun is not None:
                image = transformFun(image)
            #make output file name
            imageName = os.path.join(imagePath, videoId+'_'+\
            str(timePoints[i]).zfill(5)+'.jpg')
            #write image
            imageio.imwrite(imageName, image)


def cutMargin(image, tallSize, wideSize):
    newHeight = float(tallSize[1])/wideSize[1]*wideSize[0]
    marginTop = int(round((tallSize[0] - newHeight)/2.0))
    marginBottom = tallSize[0] - marginTop
    return(image[marginTop:marginBottom, :, :])
    
            
#set parameters
cameraSize = (720, 1280)
gazemapSize = (768, 1024)
predictionRate = 3
sampleRate = 10
#predictionRate needs to be an integer
#sampleRate needs to be an integer

videoFolder = 'data/test/camera_videos/'
imageFolder = 'data/test/camera_images/'

#make a function that cuts of the black margins of gaze maps
transformFun = lambda image: cutMargin(image, gazemapSize, cameraSize)

#parse videos
if sampleRate % predictionRate == 0:
    ParseVideos(videoFolder, imageFolder, sampleRate=sampleRate)
else:
    for i in range(predictionRate):
        ParseVideos(videoFolder, imageFolder, sampleRate=sampleRate, 
                    shift=i*1.0/sampleRate/predictionRate)
        



