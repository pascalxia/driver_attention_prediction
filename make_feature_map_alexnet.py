# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 06:37:24 2017

@author: pasca
"""

from __future__ import print_function

#from keras.applications import vgg19
import my_alexnet
import numpy as np

import data_point_collector
import BatchDatasetReader as dataset
import os
from math import ceil
import tensorflow as tf
from ut import resize_feature_map




IMAGE_SIZE = [576, 1024]
batchSize = 5
data_dir = 'data/'
data_subfolder = 'application/'


xdim = tuple(IMAGE_SIZE) + (3,)
input_tensor = tf.placeholder(tf.float32, (None,) + xdim)
feature_map = my_alexnet.AlexNet(input_tensor)



train_records, valid_records, apply_records = data_point_collector.read_datasets(data_dir)

if not os.path.isdir(data_dir+data_subfolder+'image_features_alexnet/'):
    os.mkdir(data_dir+data_subfolder+'image_features_alexnet/')

#find the images whose features were already calcualted
existing_points = [f[:-4] for f in os.listdir(data_dir+data_subfolder+'image_features_alexnet/') if f.endswith('npy')]


#exclude those data points
apply_records = list(set(apply_records) - set(existing_points))

apply_dataset_reader = dataset.BatchDataset(data_dir+data_subfolder,
                                            apply_records, 
                                            image_size=IMAGE_SIZE)




init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
sess = tf.Session(config=config)
sess.run(init)



#make feature maps for the validation dataset

nImage = len(apply_dataset_reader.data_point_names)
nBatch = int(ceil(float(nImage)/batchSize))
for j in range(nBatch):
    batch = apply_dataset_reader.next_batch(batchSize)
    images = apply_dataset_reader.get_images(batch)
    images = images.astype(np.float32)
    images[:,:,:,0] -= 123.68
    images[:,:,:,1] -= 116.79
    images[:,:,:,2] -= 103.939
    
    feed_dict = {input_tensor: images}
    featureMaps = sess.run(feature_map, feed_dict = feed_dict)
    old_size = tuple(featureMaps.shape[1:3])
    xs, ys = resize_feature_map(old_size, (34,62), pad_x=2, pad_y=2)
    resizedMaps = featureMaps[:, ys, xs, :]
    
    for i in range(len(batch)):
        np.save(data_dir+data_subfolder+'image_features_alexnet/'+batch[i]+'.npy',
                resizedMaps[i])
    print(str(j+1)+'/'+str(nBatch)+' batches for application set have been finished.')


