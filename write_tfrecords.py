import tensorflow as tf

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

import data_point_collector as dpc


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
args = parser.parse_args()


camera_folder = os.path.join(args.data_dir, 'camera_images')
feature_folder = os.path.join(args.data_dir, 'image_features_alexnet')
gazemap_folder = os.path.join(args.data_dir, 'gazemap_images')
tfrecord_folder = os.path.join(args.data_dir, 'tfrecords')

if not os.path.isdir(tfrecord_folder):
    os.makedirs(tfrecord_folder)

data_point_names = dpc.get_data_point_names(args.data_dir, in_sequences=True)

with tf.python_io.TFRecordWriter(os.path.join(tfrecord_folder, "cameras_gazes.tfrecords")) as writer:
    for seq in tqdm(data_point_names):
        camera_features = list()
        gazemap_features = list()
        for f in seq:
            camera = cv2.imread(os.path.join(camera_folder,f+'.jpg'))[:,:,[2,1,0]]
            camera = cv2.resize(camera, (1024,576), interpolation=cv2.INTER_LINEAR)
            camera_features.append(_bytes_feature(camera.tostring()))

            gazemap = cv2.imread(os.path.join(gazemap_folder, f+'.jpg'))[:,:,0]
            gazemap = cv2.resize(gazemap, (64,36), interpolation=cv2.INTER_AREA)
            gazemap_features.append(_bytes_feature(gazemap.tostring()))
        
        feature_lists = {'cameras': tf.train.FeatureList(feature=camera_features),
                         'gazemaps': tf.train.FeatureList(feature=gazemap_features)}
                         
        example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
        writer.write(example.SerializeToString())
        
with tf.python_io.TFRecordWriter(os.path.join(tfrecord_folder, "image_features_alexnet.tfrecords")) as writer:
    for seq in tqdm(data_point_names):
        feature_map_features = list()
        for f in seq:
            feature_map = np.load(os.path.join(feature_folder, f+'.npy'))
            feature_map_features.append(_bytes_feature(feature_map.tostring()))
        
        feature_lists = {'feature_maps': tf.train.FeatureList(feature=feature_map_features)}

        example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
        writer.write(example.SerializeToString())

