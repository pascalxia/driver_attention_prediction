# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 02:00:44 2017

@author: pasca
"""

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import numpy as np
import data_point_collector as dpc
import BatchDatasetReader
import scipy.misc as misc
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import moviepy.editor as mpy
import moviepy.video.io.bindings as bindings
import matplotlib.colors as colors
import argparse
import ut


#set flags--------------------------
parser = argparse.ArgumentParser()
ut.add_args_for_general(parser)
ut.add_args_for_visualization(parser)

args = parser.parse_args()
ut.parse_for_general(args)

#set parameters-------------------
gaze_map_size = (72, 128)

prediction_dir = os.path.join(args.model_dir, 'prediction_iter_'+args.model_iteration)
movie_dir = os.path.join(args.model_dir, 'visualization_prediction_iter_'+args.model_iteration)
if not os.path.isdir(movie_dir):
    os.makedirs(movie_dir)
    
if args.video_list_file is not None:
    with open(args.video_list_file, 'rt') as f:
        video_list = f.readline().split(' ')
else:
    video_list = None
    
    

#set up data reader------------------
data_points = dpc.get_data_point_names(args.data_dir, in_sequences=True, keep_prediction_rate=False)
if video_list is not None:
    data_points = dpc.keep_only_videos(data_points, video_list)
validation_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir,
                         data_points, 
                         args.image_size)

batch_size = len(validation_dataset_reader.data_point_names)

batches = validation_dataset_reader.next_batch_in_seqs(batch_size=batch_size)

#define some functions------------------------------
def prep_gazemaps_for_visual(gazemaps, desired_size, thresh):
    gazemaps_show = []
    for gazemap in gazemaps:
        gazemap = misc.imresize(gazemap, desired_size)
        if gazemap.ndim>2:
            gazemap = gazemap[:,:,0]
        gazemaps_show.append(gazemap)
    return gazemaps_show

def make_frame_for_heatmap(t, maps, cmap, fps, fig, ax, global_vmax): 
    ind = np.floor(t*fps).astype(np.int)
    if ind>=len(maps):
        return bindings.mplfig_to_npimage(fig) 
    local_vmax = maps[ind].max()
    
    ax.clear()
    ax.imshow(maps[ind], cmap=cmap, vmax=max(local_vmax, 0.5*global_vmax))
    ax.axis('off')
    return bindings.mplfig_to_npimage(fig)

#camera_images = valid_images
#predictions = prediction_show

def make_moive(camera_images, predictions):
    camera_clip = mpy.ImageSequenceClip([im for im in camera_images], fps=args.fps)
    camera_grayscale = camera_clip.fx(mpy.vfx.blackwhite)
    dpi = 100
    fig, ax = plt.subplots(1,figsize=(args.image_size[1]/dpi, args.image_size[0]/dpi), dpi=dpi)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    duration = camera_clip.duration
    
    global_vmax = predictions.max()
    
    prediction_clip = mpy.VideoClip(lambda t: \
                                    make_frame_for_heatmap(t,
                                                           predictions,
                                                           'jet', args.fps,
                                                           fig, ax, global_vmax), 
                                    duration=duration)
    prediction_clip = prediction_clip.set_opacity(args.heatmap_alpha)
                                    
    mix_clip = mpy.CompositeVideoClip([camera_grayscale, prediction_clip])
    #output_clip = mpy.clips_array([[camera_clip, mix_clip]])    
    return mix_clip

#start visualization---------------------
for batch in batches:
    print(batch)
    valid_images = validation_dataset_reader.get_images(batch)
    
    try:
        prediction_maps = [misc.imread(os.path.join(prediction_dir, data_point + '.jpg')) for data_point in batch]
    except FileNotFoundError:
        print('Did not find prediction map')
        continue
    prediction_show = prep_gazemaps_for_visual(prediction_maps, args.image_size, args.visualization_thresh)
    
    #pdb.set_trace()
    prediction_show = ut.normalize_maps(np.array(prediction_show))
    prediction_movie = make_moive(valid_images, prediction_show)
    #add text
    duration = prediction_movie.duration
    textClip = mpy.ImageClip('prediction_label.png', duration=duration)
    prediction_movie = mpy.CompositeVideoClip([prediction_movie, textClip])
    
    movie = prediction_movie
    movie_name = batch[0].split('_')[0] + '.mp4'
    
    movie.write_videofile(os.path.join(movie_dir, movie_name))


