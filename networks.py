# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:28:23 2017

@author: pasca
"""

import keras.layers as layers
import tensorflow as tf
from gaussian_smooth import GaussianSmooth
import numpy as np
import keras.layers.wrappers as wps
from keras.models import Model
from keras.applications.xception import Xception
import my_vgg19
from keras.layers.merge import concatenate
from my_squeezenet import SqueezeNet
from my_alexnet import AlexNet
from keras.layers.normalization import BatchNormalization



GAUSSIAN_KERNEL_SIZE = 15
EPSILON = 1e-12



def xception_encoder(image_size):
    encoder = Xception(include_top=False, input_shape=image_size + (3,))
    print('Model loaded.')
    
    feature_map_temp = encoder.get_layer('block13_sepconv1_act').output
    feature_net = Model(inputs=encoder.input, 
                        outputs=feature_map_temp)
    
    weight_to_monitor = feature_net.get_layer('block12_sepconv2').weights[0][0,0,0,0]
    
    return feature_net, weight_to_monitor
    

def vgg_encoder(image_size):
    encoder = my_vgg19.VGG19(weights='imagenet', include_top=False)
    print('Model loaded.')
    
    feature_map_temp = concatenate([encoder.get_layer('block5_pure_conv1').output,
                               encoder.get_layer('block5_relu1').output,
                               encoder.get_layer('block5_relu2').output,
                               encoder.get_layer('block5_pure_conv3').output,
                               encoder.get_layer('block5_relu4').output], axis=3)
    feature_net = Model(inputs=encoder.input, 
                        outputs=feature_map_temp)
    
    weight_to_monitor = feature_net.get_layer('block5_pure_conv3').weights[0][0,0,0,0]
    
    return feature_net, weight_to_monitor

def squeeze_encoder(image_size):
    encoder = SqueezeNet(input_shape=image_size + (3,))
    print('Model loaded.')
    
    feature_map_temp = encoder.get_layer('fire9/concat').output
    feature_net = Model(inputs=encoder.input, 
                        outputs=feature_map_temp)
    
    weight_to_monitor = feature_net.get_layer('fire9/expand3x3').weights[0][0,0,0,0]
    
    return feature_net, weight_to_monitor

def alex_encoder(args):
    def feature_net(input_tensor):
        feature_map = AlexNet(input_tensor)
        feature_map = tf.image.resize_nearest_neighbor(feature_map, [34,62])
        feature_map = tf.image.pad_to_bounding_box(feature_map,
                                                   0, 0,
                                                   36, 64)
        return feature_map
    
    return feature_net


def readout_net(feature_map, gaze_map_size, drop_rate, gaussian=None, gaze_prior=None):
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(2, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(1, (1, 1), activation='linear', name='readout_conv4')(x)
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)         
    
    logits = tf.reshape(x, [-1, gaze_map_size[0]*gaze_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits
        '''
        #add prior map
        with open(data_dir + 'gaze_prior.pickle', 'rb') as f:
            gaze_prior = pickle.load(f)
        if gaze_prior.shape != gaze_map_size:
            gaze_prior = misc.imresize(gaze_prior, gaze_map_size)
        gaze_prior = gaze_prior.astype(np.float32)
        gaze_prior /= np.sum(gaze_prior)
        '''
        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, gaze_map_size[0]*gaze_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits
    

def readout_big(feature_map, gaze_map_size, drop_rate, gaussian=None, gaze_prior=None):
    x = layers.Conv2D(16, (5, 5), activation='relu', 
                      name='readout_conv1', padding='same')(feature_map)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', 
                      name='readout_conv2', padding='same')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(2, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(1, (1, 1), activation='linear', name='readout_conv4')(x)
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)         
    
    logits = tf.reshape(x, [-1, gaze_map_size[0]*gaze_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits
        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, gaze_map_size[0]*gaze_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits
    
    
def readout_bn(feature_map, gaze_map_size, drop_rate, gaussian=None, gaze_prior=None):
    #batch normalization
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(2, (1, 1), activation='relu', name='readout_conv3')(x)
    x = BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(1, (1, 1), activation='linear', name='readout_conv4')(x)
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)         
    
    logits = tf.reshape(x, [-1, gaze_map_size[0]*gaze_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits
        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, gaze_map_size[0]*gaze_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits
    
    
def readout_net_BDD(feature_map, gaze_map_size, drop_rate, gaussian=None, gaze_prior=None):
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(2, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(1, (1, 1), activation='linear', name='readout_conv4')(x)
    x = layers.convolutional.UpSampling2D(size=(3, 3))(x)
    x = tf.pad(x, [[0,0], [0,0,], [0,4], [0,0]], 'CONSTANT', constant_values=0)
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)         
    
    logits = tf.reshape(x, [-1, gaze_map_size[0]*gaze_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits
        '''
        #add prior map
        with open(data_dir + 'gaze_prior.pickle', 'rb') as f:
            gaze_prior = pickle.load(f)
        if gaze_prior.shape != gaze_map_size:
            gaze_prior = misc.imresize(gaze_prior, gaze_map_size)
        gaze_prior = gaze_prior.astype(np.float32)
        gaze_prior /= np.sum(gaze_prior)
        '''
        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, gaze_map_size[0]*gaze_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits



def lstm_readout_net_old(feature_map_in_seqs, gaze_map_size, drop_rate, gaze_prior=None):
    x = wps.TimeDistributed(\
            layers.Conv2D(16, (1, 1), activation='relu', 
                          name='readout_conv1'))(feature_map_in_seqs)
    x = wps.TimeDistributed(layers.core.Dropout(drop_rate))(x)
    x = wps.TimeDistributed(\
            layers.Conv2D(32, (1, 1), activation='relu', 
                          name='readout_conv2'))(x)
    x = wps.TimeDistributed(layers.core.Dropout(drop_rate))(x)
    x = wps.TimeDistributed(\
            layers.Conv2D(2, (1, 1), activation='relu', 
                          name='readout_conv3'))(x)
    x = wps.TimeDistributed(layers.core.Dropout(drop_rate))(x)
    
    x = wps.TimeDistributed(layers.core.Reshape((-1,)))(x)

    x = layers.recurrent.LSTM(units=gaze_map_size[0]*gaze_map_size[1], 
                              dropout=drop_rate, 
                              recurrent_dropout=drop_rate,
                              return_sequences=True)(x)

    x = wps.TimeDistributed(layers.core.Dense(gaze_map_size[0]*gaze_map_size[1]))(x)
    
    x = wps.TimeDistributed(layers.core.Reshape(gaze_map_size + (1,)))(x)
    
    x = wps.TimeDistributed(\
            GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth'))(x)
    
    logits = tf.reshape(x, [-1, gaze_map_size[0]*gaze_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits

        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, gaze_map_size[0]*gaze_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits


def lstm_readout_net(feature_map_in_seqs, gaze_map_size, drop_rate, gaze_prior=None):
    batch_size = tf.shape(feature_map_in_seqs)[0]
    n_step = tf.shape(feature_map_in_seqs)[1]
    n_channel = int(feature_map_in_seqs.get_shape()[4])
    feature_map = tf.reshape(feature_map_in_seqs,  
                             [batch_size*n_step, gaze_map_size[0], 
                              gaze_map_size[1], n_channel])
    
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(2, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.core.Dropout(drop_rate)(x)
    
    #x = layers.core.Reshape((-1,))(x)
    temp_shape = x.get_shape()[1:4]
    temp_shape = [int(s) for s in temp_shape]
    x = tf.reshape(x, [batch_size, n_step, temp_shape[0]*temp_shape[1]*temp_shape[2]])

    x = layers.recurrent.LSTM(units=gaze_map_size[0]*gaze_map_size[1], 
                              dropout=drop_rate, 
                              recurrent_dropout=drop_rate,
                              return_sequences=True)(x)
    
    x = wps.TimeDistributed(layers.core.Dense(gaze_map_size[0]*gaze_map_size[1]))(x)
    
    x = tf.reshape(x, [batch_size*n_step, 
                       gaze_map_size[0], gaze_map_size[1], 1])
        
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)
    
    logits = tf.reshape(x, [-1, gaze_map_size[0]*gaze_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits

        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, gaze_map_size[0]*gaze_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits
    
    
def conv_lstm_readout_net(feature_map_in_seqs, gaze_map_size, drop_rate, gaze_prior=None):
    batch_size = tf.shape(feature_map_in_seqs)[0]
    n_step = tf.shape(feature_map_in_seqs)[1]
    n_channel = int(feature_map_in_seqs.get_shape()[4])
    feature_map = tf.reshape(feature_map_in_seqs,  
                             [batch_size*n_step, gaze_map_size[0], 
                              gaze_map_size[1], n_channel])
    
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(2, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.core.Dropout(drop_rate)(x)
    
    #x = layers.core.Reshape((-1,))(x)
    temp_shape = x.get_shape()[1:4]
    temp_shape = [int(s) for s in temp_shape]
    x = tf.reshape(x, [batch_size, n_step, temp_shape[0], temp_shape[1], temp_shape[2]])
    
    x = layers.ConvLSTM2D(filters=1,
                              kernel_size=(1,1),
                              strides=(1,1),
                              padding='same', 
                              dropout=drop_rate, 
                              recurrent_dropout=drop_rate,
                              return_sequences=True)(x)
    
    x = wps.TimeDistributed(layers.Conv2D(1, (1, 1), activation='linear'))(x)
    
    x = tf.reshape(x, [batch_size*n_step, 
                       gaze_map_size[0], gaze_map_size[1], 1])
        
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)
    
    logits = tf.reshape(x, [-1, gaze_map_size[0]*gaze_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits

        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, gaze_map_size[0]*gaze_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits
    
    
    
def big_conv_lstm_readout_net(feature_map_in_seqs, feature_map_size, drop_rate, gaze_prior=None):
    batch_size = tf.shape(feature_map_in_seqs)[0]
    n_step = tf.shape(feature_map_in_seqs)[1]
    n_channel = int(feature_map_in_seqs.get_shape()[4])
    feature_map = tf.reshape(feature_map_in_seqs,  
                             [batch_size*n_step, feature_map_size[0], 
                              feature_map_size[1], n_channel])
    
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(8, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.core.Dropout(drop_rate)(x)
    
    #x = layers.core.Reshape((-1,))(x)
    temp_shape = x.get_shape()[1:4]
    temp_shape = [int(s) for s in temp_shape]
    x = tf.reshape(x, [batch_size, n_step, temp_shape[0], temp_shape[1], temp_shape[2]])
    
    x = layers.ConvLSTM2D(filters=1,
                          kernel_size=(3,3),
                          strides=(1,1),
                          padding='same', 
                          dropout=drop_rate, 
                          recurrent_dropout=drop_rate,
                          return_sequences=True)(x)
    
    x = wps.TimeDistributed(layers.Conv2D(1, (1, 1), activation='linear'))(x)
    
    x = tf.reshape(x, [batch_size*n_step, 
                       feature_map_size[0], feature_map_size[1], 1])
        
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)
    
    logits = tf.reshape(x, [-1, feature_map_size[0]*feature_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits

        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, feature_map_size[0]*feature_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits


def thick_conv_lstm_readout_net(feature_map_in_seqs, feature_map_size, drop_rate, gaze_prior=None):
    batch_size = tf.shape(feature_map_in_seqs)[0]
    n_step = tf.shape(feature_map_in_seqs)[1]
    n_channel = int(feature_map_in_seqs.get_shape()[4])
    feature_map = tf.reshape(feature_map_in_seqs,  
                             [batch_size*n_step, feature_map_size[0], 
                              feature_map_size[1], n_channel])
    
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(8, (1, 1), activation='relu', name='readout_conv3')(x)
    #x = layers.core.Dropout(drop_rate)(x)
    
    # reshape into temporal sequence
    temp_shape = x.get_shape()[1:4]
    temp_shape = [int(s) for s in temp_shape]
    x = tf.reshape(x, [batch_size, n_step, temp_shape[0], temp_shape[1], temp_shape[2]])
    
    initial_c = layers.Conv2D(5, (3, 3), activation='tanh', padding='same')(layers.core.Dropout(drop_rate)(x[:, 0]))
    initial_c = layers.core.Dropout(drop_rate)(initial_c)
    initial_h = layers.Conv2D(5, (3, 3), activation='tanh', padding='same')(layers.core.Dropout(drop_rate)(x[:, 0]))
    initial_h = layers.core.Dropout(drop_rate)(initial_h)
    
    conv_lstm = layers.ConvLSTM2D(filters=5,
                                  kernel_size=(3,3),
                                  strides=(1,1),
                                  padding='same', 
                                  dropout=drop_rate, 
                                  recurrent_dropout=drop_rate,
                                  return_sequences=True)
    x = conv_lstm([x, initial_c, initial_h])
    
    x = wps.TimeDistributed(layers.Conv2D(1, (1, 1), activation='linear'))(x)
    
    x = tf.reshape(x, [batch_size*n_step, 
                       feature_map_size[0], feature_map_size[1], 1])
        
    x = GaussianSmooth(kernel_size = GAUSSIAN_KERNEL_SIZE, name='gaussian_smooth')(x)
    
    logits = tf.reshape(x, [-1, feature_map_size[0]*feature_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits

        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, feature_map_size[0]*feature_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits

