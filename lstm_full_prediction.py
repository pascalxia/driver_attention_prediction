# -*- coding: utf-8 -*-
"""
@author: Ye Xia
"""


from __future__ import print_function

import numpy as np
import data_point_collector
import BatchDatasetReader
import scipy.misc as misc
import tensorflow as tf
import pickle
import pdb
import re
import os
from keras import backend as K
import networks
import argparse
import ut
import pandas as pd
import feather


#set flags--------------------------
#set flags--------------------------
parser = argparse.ArgumentParser()
ut.add_args_for_general(parser)
ut.add_args_for_inference(parser)
ut.add_args_for_evaluation(parser)
ut.add_args_for_full(parser)
ut.add_args_for_lstm(parser)


args = parser.parse_args()
ut.parse_for_general(args)


#set parameters-------------------
KERNEL_SIZE = 15

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
annotation_in_seqs = tf.placeholder(tf.float32, shape=(None, None) + args.gaze_map_size + (1,), 
                                                                        name="annotation")


#set up encoder net-----------------
input_tensor = tf.reshape(tf.cast(input_image_in_seqs, tf.float32),    
                                                    [-1, args.image_size[0], args.image_size[1], 3])
input_tensor = input_tensor - [123.68, 116.79, 103.939]


with tf.variable_scope("encoder"):
    if args.encoder == 'vgg':
    feature_net, weight_to_monitor = networks.vgg_encoder(args.image_size)
elif args.encoder == 'squeeze':
    feature_net, weight_to_monitor = networks.squeeze_encoder(args.image_size)
elif args.encoder == 'xception':
    feature_net, weight_to_monitor = networks.xception_encoder(args.image_size)
elif args.encoder == 'alex':
    feature_net = networks.alex_encoder(args)
else:
    print('The entered encoder is wrong.')
exit
feature_map = feature_net(input_tensor)


#set up readout net----------------------------
batch_size_tensor = tf.shape(input_image_in_seqs)[0]
n_steps_tensor = tf.shape(input_image_in_seqs)[1]
feature_map_size = (int(feature_map.get_shape()[1]), 
                                        int(feature_map.get_shape()[2]))
n_channel = int(feature_map.get_shape()[3])

with tf.variable_scope("readout"):
    if args.readout=='default':
    readout_net = networks.lstm_readout_net
elif args.readout=='conv_lstm':
    readout_net = networks.conv_lstm_readout_net
elif args.readout=='big_conv_lstm':
    readout_net = networks.big_conv_lstm_readout_net

feature_map_in_seqs = tf.reshape(feature_map,
                                                                 [batch_size_tensor, n_steps_tensor,
                                                                    feature_map_size[0], feature_map_size[1],
                                                                    n_channel])
if args.use_prior is True:        
    #load prior map
    with open(args.data_dir + 'gaze_prior.pickle', 'rb') as f:
    gaze_prior = pickle.load(f)
if gaze_prior.shape != args.gaze_map_size:
    gaze_prior = misc.imresize(gaze_prior, args.gaze_map_size)
gaze_prior = gaze_prior.astype(np.float32)
gaze_prior /= np.sum(gaze_prior)
logits, pre_prior_logits = \
readout_net(feature_map_in_seqs, feature_map_size, 
                        args.drop_rate, gaze_prior)
pre_prior_annotation = tf.nn.softmax(pre_prior_logits)
else:
    logits = readout_net(feature_map_in_seqs, 
                                             feature_map_size, args.drop_rate)
pre_prior_annotation = None

#predicted annotation
pred_annotation = tf.nn.softmax(logits)


#set up loss---------------------------------------
loss, accuracy_loss, reg_loss, spread, kls = \
ut.set_losses(logits, 
                            pred_annotation,
                            annotation_in_seqs,
                            args)

#set up data readers-------------------------------
_, valid_data_points = \
data_point_collector.read_datasets_in_sequences(args.data_dir)
validation_dataset_reader = \
BatchDatasetReader.BatchDataset(args.data_dir+'validation/',
                                                                valid_data_points, 
                                                                args.image_size)


#initialize variables except for encoder variables------------------
encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                                                 scope='encoder')
vars_to_init = list(set(tf.global_variables()) - set(encoder_vars))
sess.run(tf.variables_initializer(vars_to_init))
if args.encoder == 'alex':
    #also initialize the encoder variables
    sess.run(tf.variables_initializer(encoder_vars))


#set up savers------------
saver = tf.train.Saver(var_list=vars_to_init, max_to_keep=20)


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
validation_losses = []
n_iteration = np.ceil(len(
    validation_dataset_reader.data_point_names)/args.batch_size).astype(np.int)

dir_name = args.model_dir+'prediction_iter_'+args.model_iteration+'/'
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

dfs = []
for itr in range(n_iteration):
    print('Doing iteration %d/%d' % (itr, n_iteration))
batch = validation_dataset_reader.next_batch_in_seqs(batch_size=args.batch_size)
valid_input_images = validation_dataset_reader.get_images_in_seqs(batch)
valid_annotations = validation_dataset_reader.\
get_annotations_in_seqs(batch, desired_size=args.gaze_map_size)

feed_dict = {input_image_in_seqs: valid_input_images, 
    annotation_in_seqs: valid_annotations,
    K.learning_phase(): 0}
[prediction, valid_loss, valid_kls] = sess.run([pred_annotation, loss, kls], 
                                                                                             feed_dict=feed_dict)
#flatten batch
flat_batch = [data_point for video in batch for data_point in video]
for i in range(len(prediction)):
    #save predicted map
    prediction_map = prediction[i].reshape(args.gaze_map_size)
fpath = dir_name + flat_batch[i] + '.jpg'
misc.imsave(fpath, prediction_map)

#save kl-divergences
df = pd.DataFrame.from_dict({
    'fileName': flat_batch,
    'kl': valid_kls})
dfs.append(df)

#save loss
validation_losses.append(valid_loss)

kl_df = pd.concat(dfs)
feather.write_dataframe(kl_df, dir_name+'kls.feather')

mean_valid_loss = np.mean(validation_losses)
with open(dir_name+"mean_validation_loss_%f.txt" % mean_valid_loss, "w") as text_file:
    text_file.write("Mean validation loss: %f" % mean_valid_loss)
print('Mean validation loss is %f' % mean_valid_loss)

