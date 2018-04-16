# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 23:49:39 2017

@author: pasca
"""
from __future__ import print_function
import os
import imageio
import numpy as np
from tqdm import tqdm
import argparse


def parse_videos(video_dir, image_dir, parse_rate, transform_fn=None, 
                overwrite=False, shift=0, video_suffix='.mp4'):
    # parse_rate is how many Hz the videos should be parsed
    # shift is in seconds, means starting parsing in 'shift' seconds
    # overwrite=False means if some videos have already had parsed frames in image_dir, then skip these videos
    
    # make sure output directory is there
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    
    if not overwrite:
        # collect already parsed videos
        old_video_ids = [f.split('_')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    
    video_names = [f for f in os.listdir(video_dir) if f.endswith(video_suffix)]
    for file in tqdm(video_names):
        filename = os.path.join(video_dir, file)
        print(filename)
        video_id = file.split('.')[0]
        if not overwrite:
            # skip parsed videos
            if video_id in old_video_ids:
                print('skip')
                continue
        
        try:
            reader = imageio.get_reader(filename)
        except OSError:
            with open("video_parsing_errors.txt", "a") as myfile:
                myfile.write(video_id+'\n')
            continue
        
        fps = reader.get_meta_data()['fps']
        duration = reader.get_meta_data()['duration']
        n_frames = reader.get_meta_data()['nframes']
        
        
        if parse_rate is not None:
            # calculate the time points in ms to sample frames
            time_points = np.arange(shift*1000, duration*1000, 1000.0/parse_rate)
            time_points = np.floor(time_points).astype(int)
            # calculate the frame indexes
            frame_indexes = (np.floor(time_points/1000.0*fps)).astype(int)
            sample_size = len(frame_indexes)
        else:
            frame_indexes = np.arange(n_frames)
            time_points = (frame_indexes*1000/fps).astype(int)
            sample_size = n_frames
        
        for i in tqdm(range(sample_size)):
            # make output file name
            image_name = os.path.join(image_dir, video_id+'_'+\
                str(time_points[i]).zfill(5)+'.jpg')
                
            if os.path.isfile(image_name):
                print('Already exist.')
                continue
          
            # read image
            try:
                image = reader.get_data(frame_indexes[i])
            except:
                print('Can\'t read this frame. Skip')
                continue
            
            # apply transformation
            if transform_fn is not None:
                image = transform_fn(image)
            
            # write image
            imageio.imwrite(image_name, image)
        
    

def main(args):
    # parse videos
    if args.sample_rate % args.prediction_rate == 0:
        parse_videos(args.video_dir, args.image_dir, 
            parse_rate=args.sample_rate, video_suffix=args.video_suffix)
    else:
        parse_videos(args.video_dir, args.image_dir, 
            parse_rate=args.sample_rate*args.prediction_rate,
            video_suffix=args.video_suffix)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',
        type=str,
        default='data/application/camera_videos',
        help='the directory that contains videos to parse')
    parser.add_argument('--image_dir',
        type=str,
        default='data/application/camera_images',
        help='the directory of parsed frame images')
    parser.add_argument('--sample_rate',
        type=int,
        default=3,
        help='at how many Hz the attention prediction results are needed')
    parser.add_argument('--prediction_rate',
        type=int,
        default=3,
        help='at how many Hz will the network predicts attention maps')
    parser.add_argument('--video_suffix',
        type=str,
        default='.mp4',
        help='the suffix of video files. E.g., .mp4')
    
    args = parser.parse_args()
    
    main(args)
    

