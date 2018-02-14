# -*- coding: utf-8 -*-
"""
@author: Ye Xia
"""

import numpy as np
import data_point_collector
import BatchDatasetReader
import scipy.misc as misc
import tensorflow as tf
import pickle
import re
import os
from keras import backend as K
import networks
import argparse
import ut
import pandas as pd
import feather



#set flags--------------------------
parser = argparse.ArgumentParser()
ut.add_args_for_general(parser)
ut.add_args_for_inference(parser)
ut.add_args_for_evaluation(parser)
ut.add_args_for_feature(parser)
ut.add_args_for_lstm(parser)

args = parser.parse_args()
ut.parse_for_general(args)
ut.parse_for_feature(args)


#set parameters-------------------
args.epsilon = 1e-12
args.gaze_map_size = (36, 64)


#set up session------------------
if args.gpu_memory_fraction is not None:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    sess = tf.Session(config=config)
else:
    sess = tf.Session()
#assign session for Keras
K.set_session(sess)


#set up placeholders---------------------------
feature_map_in_seqs = tf.placeholder(tf.float32, shape=(None, None) + args.feature_map_size + (args.feature_map_channels,), 
                             name="feature_map_in_seqs")


#set up readout net-----------------
if args.readout=='default':
    readout_net = networks.lstm_readout_net
elif args.readout=='conv_lstm':
    readout_net = networks.conv_lstm_readout_net
elif args.readout=='big_conv_lstm':
    readout_net = networks.big_conv_lstm_readout_net

if args.use_prior is True:    
    #load prior map
    with open(args.data_dir + 'gaze_prior.pickle', 'rb') as f:
        gaze_prior = pickle.load(f)
    if gaze_prior.shape != args.gaze_map_size:
        gaze_prior = ut.resize_distribution(gaze_prior, args.gaze_map_size)
    gaze_prior = gaze_prior.astype(np.float32)
    gaze_prior /= np.sum(gaze_prior)
    logits, pre_prior_logits = \
        readout_net(feature_map_in_seqs, args.feature_map_size, args.drop_rate, gaze_prior)
    pre_prior_annotation = tf.nn.softmax(pre_prior_logits)
else:
    logits = readout_net(feature_map_in_seqs, args.feature_map_size, args.drop_rate)

#predicted annotation
pred_annotation = tf.nn.softmax(logits)


#set up data readers-------------------------------
_, _, apply_data_points = \
    data_point_collector.read_datasets(args.data_dir, in_sequences=True)
apply_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir+'application/',
                         apply_data_points, 
                         args.image_size,
                         feature_name=args.feature_name)


#set up savers------------
saver = tf.train.Saver()


#try to reload weights--------------------
ckpt = tf.train.get_checkpoint_state(args.model_dir)
#pdb.set_trace()
if ckpt and ckpt.model_checkpoint_path:
    if args.model_iteration is not None:
        ckpt_path = re.sub('(ckpt-)[0-9]+', r'\g<1>'+args.model_iteration, ckpt.model_checkpoint_path)
    else:
        ckpt_path = ckpt.model_checkpoint_path
        args.model_iteration = re.search('ckpt-([0-9]+)', ckpt_path).group(1)
    saver.restore(sess, ckpt_path)
    print("Model restored...")
    
    
#start predicting-------------------------
n_iteration = np.ceil(len(
    apply_dataset_reader.data_point_names)/args.batch_size).astype(np.int)

dir_name = args.model_dir+'prediction_iter_'+args.model_iteration+'/'
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
    

for itr in range(n_iteration):
    print('Doing iteration %d/%d' % (itr, n_iteration))
    batch = apply_dataset_reader.next_batch_in_seqs(batch_size=args.batch_size)
    apply_feature_maps = apply_dataset_reader.get_feature_maps_in_seqs(batch)
    
    feed_dict = {feature_map_in_seqs: apply_feature_maps, 
                 K.learning_phase(): 0}
    prediction = sess.run(pred_annotation, 
                          feed_dict=feed_dict)
    #flatten batch
    flat_batch = [data_point for video in batch for data_point in video]
    for i in range(len(prediction)):
        #save predicted map
        prediction_map = prediction[i].reshape(args.gaze_map_size)
        fpath = dir_name + flat_batch[i] + '.jpg'
        misc.imsave(fpath, prediction_map)


