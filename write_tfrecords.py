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

for seq in tqdm(data_point_names):
    with tf.python_io.TFRecordWriter(tfrecord_folder+seq[0]+'.tfrecords') as writer:
        for f in seq:
            camera = cv2.imread(camera_folder+f+'.jpg')[:,:,[2,1,0]]
            feature_map = np.load(feature_folder+f+'.npy')
            gazemap = cv2.imread(gazemap_folder+f+'.jpg')[:,:,0]
            feature = {'camera': _bytes_feature(camera.tostring()),
                       'feature_map': _bytes_feature(feature_map.tostring()),
                       'gazemap': _bytes_feature(gazemap.tostring())}
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            writer.write(example.SerializeToString())
