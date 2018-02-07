# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 20:31:19 2017

@author: pasca
"""


from __future__ import print_function

import data_point_collector
import BatchDatasetReader
import tensorflow as tf
import datetime
import numpy as np
import pickle
from keras import backend as K
import networks
from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
import argparse
import ut


#set flags--------------------------
parser = argparse.ArgumentParser()
ut.add_args_for_general(parser)
ut.add_args_for_inference(parser)
ut.add_args_for_training(parser)
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
input_image_in_seqs = tf.placeholder(tf.uint8, shape=(None, None, args.image_size[0], args.image_size[1], 3), 
                             name="input_image")
feature_map_in_seqs = tf.placeholder(tf.float32, shape=(None, None) + args.feature_map_size + (args.feature_map_channels,), 
                             name="feature_map_in_seqs")
annotation_in_seqs = tf.placeholder(tf.float32, shape=(None, None) + args.gaze_map_size + (1,), 
                            name="annotation")


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
        readout_net(feature_map_in_seqs, args.gaze_map_size, 
                                  args.drop_rate, gaze_prior)
    pre_prior_annotation = tf.nn.softmax(pre_prior_logits)
else:
    logits = readout_net(feature_map_in_seqs, 
                                       args.gaze_map_size, args.drop_rate)
    pre_prior_annotation = None

#predicted annotation
pred_annotation = tf.nn.softmax(logits)


#set up losses----------------------------------
loss, accuracy_loss, reg_loss, spread, _ = \
    ut.set_losses(logits, 
                  pred_annotation,
                  annotation_in_seqs,
                  args)


#set up summaries------------------------------
quick_summary_op, slow_summary_op, valid_summary_op = \
    ut.make_summaries(input_image_in_seqs, 
                      pre_prior_annotation,
                      pred_annotation,
                      annotation_in_seqs,
                      loss, accuracy_loss, reg_loss, spread,
                      args)


#set up training op-------------------------
train_op = tf.train.AdamOptimizer(learning_rate = args.learning_rate).minimize(loss)


#set up data readers-------------------------------
train_data_points, valid_data_points = \
    data_point_collector.read_datasets_in_sequences(args.data_dir)
train_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir+'training/',
                         train_data_points, 
                         args.image_size,
                         feature_name=args.feature_name,
                         weight_data=args.weight_data,
                         annotation_threshold=args.annotation_threshold)
validation_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir+'validation/',
                         valid_data_points, 
                         args.image_size,
                         feature_name=args.feature_name,
                         annotation_threshold=args.annotation_threshold)


#initialize variables------------------
sess.run(tf.global_variables_initializer())
    

#set up savers------------
saver = tf.train.Saver(max_to_keep=20)
summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)




#try to reload weights--------------------
if args.model_dir is not None:
    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        vars_stored = [var[0] for var in list_variables(args.model_dir)]
        vars_restore = [v for v in tf.global_variables() if v.name[0:-2] in vars_stored]
        restore_saver = tf.train.Saver(vars_restore)
        restore_saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("Model restore failed...")



#start training-------------------------
for itr in range(args.max_iteration):
    if not args.weight_data:
        batch = train_dataset_reader.next_batch_in_seqs(args.batch_size, args.n_steps)
    else:
        batch = train_dataset_reader.random_batch_in_seqs(args.batch_size, args.n_steps)
    train_input_images = train_dataset_reader.get_images_in_seqs(batch)
    train_feature_maps = train_dataset_reader.get_feature_maps_in_seqs(batch)
    train_annotations = train_dataset_reader.\
        get_annotations_in_seqs(batch,
                                desired_size = args.gaze_map_size)
    
    feed_dict = {input_image_in_seqs: train_input_images,
                 feature_map_in_seqs: train_feature_maps, 
                 annotation_in_seqs: train_annotations,
                 K.learning_phase(): 1}
    sess.run(train_op, feed_dict=feed_dict)

    if itr % args.quick_summary_period == 0:
        train_loss, train_spread, summary_str = sess.run([loss, spread, quick_summary_op], 
                                           feed_dict=feed_dict)
        print("Step: %d, Train_loss:%g, Spread:%g" % (itr, train_loss, train_spread))
        summary_writer.add_summary(summary_str, itr)
    
    if itr % args.slow_summary_period == 0:
        summary_str = sess.run(slow_summary_op, 
                               feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, itr)

    if itr % args.valid_summary_period == 0:
        batch = validation_dataset_reader.random_batch_in_seqs(\
            args.batch_size*args.valid_batch_factor, args.n_steps)
        #pdb.set_trace()
        validation_input_images = validation_dataset_reader.get_images_in_seqs(batch)
        validation_feature_maps = validation_dataset_reader.get_feature_maps_in_seqs(batch)
        validation_annotations = \
            validation_dataset_reader.\
            get_annotations_in_seqs(batch, desired_size = args.gaze_map_size)            
            
        [validation_loss, summary_str] = sess.run([loss, valid_summary_op], 
                                   feed_dict={input_image_in_seqs: validation_input_images,
                                              feature_map_in_seqs: validation_feature_maps, 
                                              annotation_in_seqs: validation_annotations,
                                              K.learning_phase(): 0})
        summary_writer.add_summary(summary_str, itr)
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), validation_loss))
        saver.save(sess, args.logs_dir + "model.ckpt", itr)

        



