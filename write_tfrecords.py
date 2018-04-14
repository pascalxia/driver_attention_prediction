import tensorflow as tf

import os
import numpy as np
import cv2
from tqdm import tqdm

import data_point_collector as dpc


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

data_folder = 'example_data/application/'
camera_folder = 'example_data/application/camera_images/'
feature_folder = 'example_data/application/image_features_alexnet/'
gazemap_folder = 'example_data/application/gazemap_images/'
tfrecord_folder = 'example_data/tfrecords/'

data_point_names = dpc.get_data_point_names(data_folder, in_sequences=True)

with tf.python_io.TFRecordWriter(tfrecord_folder+"camera_gaze.tfrecords") as writer:
    for seq in tqdm(data_point_names):
        camera      = list()
        gazemap     = list()
        for f in seq:
            img = cv2.imread(camera_folder+f+'.jpg')[:,:,[2,1,0]]
            img = cv2.resize(img, (1024,576), interpolation=cv2.INTER_LINEAR)
            camera     .append(img)

            img = cv2.imread(gazemap_folder+f+'.jpg')[:,:,0]
            img = cv2.resize(img, (64,36), interpolation=cv2.INTER_AREA)
            gazemap    .append(img)
        
        camera      = np.array(camera)
        gazemap     = np.array(gazemap)
        feature = { 'camera':      _bytes_feature(camera.tostring()),
                    'gazemap':     _bytes_feature(gazemap.tostring()) }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        
with tf.python_io.TFRecordWriter(tfrecord_folder+"image_features_alexnet.tfrecords") as writer:
    for seq in tqdm(data_point_names):
        feature_map = list()
        for f in seq:
            feature_map.append(np.load(feature_folder+f+'.npy'))

        feature_map = np.array(feature_map)

        feature = { 'feature_map': _bytes_feature(feature_map.tostring()) }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

