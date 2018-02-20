# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 00:46:35 2017

@author: pasca
"""
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import scipy.ndimage.filters as filters
import moviepy.editor as mpy


##set up argument parser--------------------
def add_args(args, parser):
    for d in args:
        if 'required' in d:
            parser.add_argument('--'+d['name'],
                                default=d['default'],
                                type=d['type'],
                                help=d['help'],
                                required=d['required'])
        else:
            parser.add_argument('--'+d['name'],
                                default=d['default'],
                                type=d['type'],
                                help=d['help'])

def add_args_for_general(parser):
    args = [
    {
     'name': 'data_dir',
     'default': 'data/',
     'type': str,
     'help': 'folder of dataset'},
    {
     'name': 'model_dir',
     'default': None,
     'type': str,
     'help': 'folder from which restore the model '},
    {
     'name': 'image_size',
     'default': '576,1024',
     'type': str,
     'help': 'Size of the input image'}
    ]
    add_args(args, parser)

def parse_for_general(args):
    args.image_size = tuple([int(num) for num in args.image_size.split(',')])

def add_args_for_inference(parser):
    args = [
    {
     'name': 'batch_size',
     'default': 20,
     'type': int,
     'help': 'basic batch size'},
    {
     'name': 'use_prior',
     'default': False,
     'type': bool,
     'help': 'whether to use prior gaze map'},
    {
     'name': 'drop_rate',
     'default': 0,
     'type': float,
     'help': 'drop rate'},
    {
     'name': 'readout',
     'default': 'default',
     'type': str,
     'help': 'which readout network to use'},
    {
     'name': 'sparsity_weight',
     'default': 0,
     'type': float,
     'help': 'The weight of sparsity regularization'}, 
    {
     'name': 'gpu_memory_fraction',
     'default': None,
     'type': float,
     'help': 'The fraction of GPU memory to use'},
     {
     'name': 'binary',
     'default': False,
     'type': bool,
     'help': 'Whether to make the gaze maps to binary maps'},
     {
     'name': 'annotation_threshold',
     'default': None,
     'type': float,
     'help': 'When the gaze density is more than annotation_threshold times the uniform density, the pixel is gazed'}
    ]
    add_args(args, parser)
    
    
def add_args_for_feature(parser):
    args = [
    {
     'name': 'feature_name',
     'default': 'vgg',
     'type': str,
     'help': 'Which kind of features to use'},
    {
     'name': 'feature_map_size',
     'default': None,
     'type': str,
     'help': 'Feature map size (not include the number of channels)'},
    {
     'name': 'feature_map_channels',
     'default': 2560,
     'type': int,
     'help': 'The number of feature map channels'}
    ]
    add_args(args, parser)
    
def parse_for_feature(args):
    if args.feature_map_size is None:
        args.feature_map_size = (int(args.image_size[0]/16), int(args.image_size[1]/16))
    else:
        args.feature_map_size = tuple([int(num) for num in args.feature_map_size.split(',')])
        
    
def add_args_for_full(parser):
    args = [
    {
     'name': 'encoder',
     'default': 'vgg',
     'type': str,
     'help': 'Which encoder to use'}
    ]
    add_args(args, parser)
    
    
def add_args_for_training(parser):
    args = [
    {
     'name': 'learning_rate',
     'default': 1e-3,
     'type': float,
     'help': 'Learning rate for Adam Optimizer'},
    {
     'name': 'max_iteration',
     'default': 10001,
     'type': int,
     'help': 'Maximum iterations'},
    {
     'name': 'quick_summary_period',
     'default': 10,
     'type': int,
     'help': 'After how many iterations do some quick summaries'},
    {
     'name': 'slow_summary_period',
     'default': 50,
     'type': int,
     'help': 'After how many iterations do some slow summaries'},
    {
     'name': 'valid_summary_period',
     'default': 500,
     'type': int,
     'help': 'After how many iterations do validation and save one checkpoint'},
    {
     'name': 'valid_batch_factor',
     'default': 2,
     'type': int,
     'help': 'The batch size for validation is equal to this number multiply the original batch size'},
    {
     'name': 'logs_dir',
     'default': None,
     'type': str,
     'help': 'path to logs directory',
     'required': True},
    {
     'name': 'weight_data',
     'default': False,
     'type': bool,
     'help': 'whether to weight the data points differently in trianing'}
    ]
    add_args(args, parser)
    
    
def add_args_for_evaluation(parser):
    args = [
    {
     'name': 'model_iteration',
     'default': None,
     'type': str,
     'help': 'The model of which iteration to resotre'}
    ]
    add_args(args, parser)
    
    
def add_args_for_visualization(parser):
    args = [
    {
     'name': 'model_iteration',
     'default': None,
     'type': str,
     'help': 'The model of which iteration to restore'},
    {
     'name': 'visualization_thresh',
     'default': 1e-5,
     'type': float,
     'help': 'Probability density threshold for visualization'},
    {
     'name': 'video_list_file',
     'default': None,
     'type': str,
     'help': 'A txt file that contains the list of the videos to visualize, seperated by space'},
    {
     'name': 'fps',
     'default': 3,
     'type': float,
     'help': 'Frames per second'},
    {
     'name': 'heatmap_alpha',
     'default': 0.5,
     'type': float,
     'help': 'Transparency for heat map. 1 is fully opaque.'},
    {
     'name': 'turing_area_table',
     'default': None,
     'type': str,
     'help': 'Path to the table that stores the highlighted areas of Turing GT videos.'},
    {
     'name': 'skip_first_n_frames',
     'default': None,
     'type': int,
     'help': 'Number of frames to skip in the beginning.'}  
    ]
    add_args(args, parser)


def add_args_for_lstm(parser):
    args = [
    {
     'name': 'n_steps',
     'default': None,
     'type': int,
     'help': 'number of time steps for each sequence'},
     {
     'name': 'longest_seq',
     'default': None,
     'type': int,
     'help': 'How many frames can the longest sequence contain'}
    ]
    add_args(args, parser)


##set up summaries---------------------------
def make_summaries(input_image, 
                   pre_prior_annotation,
                   pred_annotation,
                   annotation,
                   loss, accuracy_loss, reg_loss, spread,
                   args):
    #quick summaries
    quick_summaries = []
    quick_summaries.append(tf.summary.scalar("training_loss", loss))
    quick_summaries.append(tf.summary.scalar("accuracy_loss", accuracy_loss))
    quick_summaries.append(tf.summary.scalar("sparsity_loss", reg_loss))
    quick_summaries.append(tf.summary.scalar('prediction_min', 
                                                tf.reduce_min(pred_annotation)))
    quick_summaries.append(tf.summary.scalar('prediction_max', 
                                                tf.reduce_max(pred_annotation)))
    quick_summaries.append(tf.summary.scalar("training_spread", spread))
    quick_summary_op = tf.summary.merge(quick_summaries)
    
    
    #slow summaries
    #input image summary
    slow_summaries = []
    resized_input_image = tf.reshape(input_image, [-1, args.image_size[0], args.image_size[1], 3])
    slow_summaries.append(tf.summary.image("input_image", 
                                          resized_input_image, max_outputs=2))
    
    #before prior summary
    if args.use_prior is True:
        prior_image = tf.reshape(tensor=pre_prior_annotation,
                                 shape=(-1, args.gaze_map_size[0], args.gaze_map_size[1], 1))
        prior_summary = tf.summary.image("before_prior", prior_image, max_outputs=2)
        slow_summaries.append(prior_summary)
        slow_summaries.append(tf.summary.histogram('before_prior', pre_prior_annotation))
    
    #prediction summary
    pred_image = tf.reshape(tensor=pred_annotation,
                            shape=(-1, args.gaze_map_size[0], args.gaze_map_size[1], 1))
    slow_summaries.append(tf.summary.image("pred_annotation", 
                                          pred_image, max_outputs=2))
    slow_summaries.append(tf.summary.histogram('pred_annotation', pred_annotation))
    
    #ground truth summary
    resized_annotation = tf.reshape(annotation, 
                                    [-1, args.gaze_map_size[0], args.gaze_map_size[1], 1])
    slow_summaries.append(tf.summary.image("annotation", 
                                          resized_annotation, max_outputs=2))
    
    slow_summary_op = tf.summary.merge(slow_summaries)
    
    
    #summaries for validation
    valid_summaries = []
    valid_summaries.append(tf.summary.scalar("validation_loss", loss))
    valid_summaries.append(tf.summary.scalar("validation_accuracy", accuracy_loss))
    valid_summaries.append(tf.summary.scalar("validation_sparsity", reg_loss))
    valid_summaries.append(tf.summary.scalar("validation_spread", spread))
    valid_summaries.append(tf.summary.image("validation_input_image", 
                                          resized_input_image, max_outputs=2))
    valid_summaries.append(tf.summary.image("validation_pred_annotation", 
                                          pred_image, max_outputs=2))
    valid_summaries.append(tf.summary.image("validation_annotation", 
                                          resized_annotation, max_outputs=2))
    valid_summary_op = tf.summary.merge(valid_summaries)
    
    return quick_summary_op, slow_summary_op, valid_summary_op


##set up losses---------------------------------
def set_losses(logits, 
               pred_annotation,
               annotation,
               args):

    y = tf.reshape(annotation, [-1, args.gaze_map_size[0]*args.gaze_map_size[1]])
    if args.binary is not True:
        accuracy_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y)
    else:
        accuracy_losses = tf.reduce_mean(\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                    labels=y),
            axis=(1,))
    
    if args.binary is not True:
        y_ = tf.maximum(y, args.epsilon)
        p_entropies = tf.reduce_sum(-tf.multiply(y_, tf.log(y_)), axis=1)
        kls = accuracy_losses - p_entropies
    else:
        kls = accuracy_losses
    
    accuracy_loss = tf.reduce_mean(accuracy_losses)
    
    before_log = tf.maximum(pred_annotation, tf.constant(args.epsilon))
    reg_loss = tf.reduce_mean(tf.reduce_sum(-tf.multiply(before_log, 
                                                    tf.log(before_log)), 
                                       axis=1))
    reg_loss = tf.maximum(reg_loss, tf.constant(4.5))
    
    grid_x, grid_y = np.meshgrid(np.arange(args.gaze_map_size[1]), 
                                 np.arange(args.gaze_map_size[0]))
    dist_matrix = np.sqrt(np.square(grid_x-args.gaze_map_size[1]/2) + \
                          np.square(grid_y-args.gaze_map_size[0]/2))
    spread = tf.multiply(pred_annotation, np.reshape(dist_matrix, (-1,)))
    spread = tf.reduce_mean(tf.reduce_sum(spread, axis=1))
    
    
    if args.sparsity_weight is not None:
        loss = accuracy_loss + args.sparsity_weight * reg_loss
    else:
        loss = accuracy_loss
        
    return loss, accuracy_loss, reg_loss, spread, kls

def resize_distribution(dist_image, target_size):
    #when downsizing
    if dist_image.shape[0] > target_size[0]:
        dist_image = misc.imresize(dist_image, target_size, interp='bilinear')
    #when upsizeing
    elif dist_image.shape[0] < target_size[0]:
        dist_image = misc.imresize(dist_image, target_size, interp='nearest')
    return dist_image


def normalize_maps(maps): 
    normalized = np.zeros(maps.shape)
    for i in range(len(maps)):
        frame_sum = np.sum(maps[i])
        if frame_sum != 0:
            normalized[i] = maps[i]/frame_sum
        else:
            normalized[i] = 1/normalized[i].size
    return normalized

def normalize_map(a_map): 
    frame_sum = np.sum(a_map)
    if frame_sum != 0:
        normalized = a_map.astype(float)/frame_sum
    else:
        normalized = np.ones(a_map.shape)/a_map.size
    return normalized


def make_turing_moive(camera_images, gazemaps, thresh, fps):
    if len(gazemaps.shape)==3:
        gazemaps = np.expand_dims(gazemaps, axis=-1)
    blurred_images = filters.gaussian_filter(camera_images, sigma=(0,5,5,0))
    
    camera_clip = mpy.ImageSequenceClip([im for im in camera_images], fps=fps)
    blurred_clip = mpy.ImageSequenceClip([im for im in blurred_images], fps=fps)
    
    masks = (gazemaps>thresh).astype(np.float)*255
    blurred_masks = filters.gaussian_filter(masks, sigma=(0,2,2,0))
    mask_clip = mpy.ImageSequenceClip(list(blurred_masks), fps=fps).to_mask()
    fovea_clip = camera_clip.set_mask(mask_clip)
    mix_clip = mpy.CompositeVideoClip([blurred_clip, fovea_clip])
    return mix_clip

def resize_feature_map(old_size, new_size, pad_x, pad_y):    
    #scaling
    x = np.linspace(-0.5, old_size[1]-0.5, new_size[1]+1)
    x = x[:-1] + old_size[1]/new_size[1]/2
    x = np.round(x).astype(int)
    #padding
    x = np.concatenate((x, np.repeat(x[-1], pad_x)))
    
    #scaling
    y = np.linspace(-0.5, old_size[0]-0.5, new_size[0]+1)
    y = y[:-1] + old_size[0]/new_size[0]/2
    y = np.round(y).astype(int)
    #padding
    y = np.concatenate((y, np.repeat(y[-1], pad_y)))
    
    xs, ys = np.meshgrid(x, y)
    #new_map = old_map[ys, xs, :]
    
    return xs, ys
