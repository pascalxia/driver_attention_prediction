import tensorflow as tf

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import random


import data_point_collector as dpc


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--n_divides', type=int, default=1)
parser.add_argument('--feature_name', type=str, default='alexnet')
args = parser.parse_args()


camera_folder = os.path.join(args.data_dir, 'camera_images')
feature_folder = os.path.join(args.data_dir, 'image_features_'+args.feature_name)
gazemap_folder = os.path.join(args.data_dir, 'gazemap_images')
tfrecord_folder = os.path.join(args.data_dir, 'tfrecords')

if not os.path.isdir(tfrecord_folder):
    os.makedirs(tfrecord_folder)

data_point_names = dpc.get_data_point_names(args.data_dir, in_sequences=True)

random.shuffle(data_point_names)
splits = [[] for _ in range(args.n_divides)]
for i in range(len(data_point_names)):
    splits[i%args.n_divides].append(data_point_names[i])

for i in range(len(splits)):
    with tf.python_io.TFRecordWriter(
        os.path.join(tfrecord_folder, 
        "cameras_gazes_%s_features_%d.tfrecords" % (args.feature_name, i) )) as writer:
        
        for seq in tqdm(splits[i]):
            camera_features = list()
            feature_map_features = list()
            gazemap_features = list()
            gaze_ps_features = list()
            for f in seq:
                # write camera images
                camera = cv2.imread(os.path.join(camera_folder,f+'.jpg'))               # do not flip bgr for imencode
                camera = cv2.resize(camera, (1024,576), interpolation=cv2.INTER_LINEAR) # please check if this is the desired size
                camera = cv2.imencode('.jpg', camera)[1].tostring()                     # imencode returns tuple(bool, ndarray)
                camera_features.append(camera)
                
                # write image feature maps
                feature_map = np.load(os.path.join(feature_folder, f+'.npy'))
                feature_map_features.append(_bytes_feature(feature_map.tostring()))
                
                # write gaze probability distribution
                gazemap = cv2.imread(os.path.join(gazemap_folder, f+'.jpg'))[:,:,0]
                gaze_ps = cv2.resize(gazemap, (64,36), interpolation=cv2.INTER_AREA)
                gaze_ps = gaze_ps.reshape((64*36,))
                gaze_ps = gaze_ps/np.sum(gaze_ps)
                gaze_ps_features.append(_bytes_feature(gaze_ps.tostring()))
                
                # write gazemap images
                gazemap = cv2.resize(gazemap, (1024,576), interpolation=cv2.INTER_AREA) # please check this size as well
                gazemap = cv2.imencode('.jpg', gazemap)[1].tostring()
                gazemap_features.append(gazemap)
            
            feature_lists = {'feature_maps': tf.train.FeatureList(feature=feature_map_features),
                             'gaze_ps': tf.train.FeatureList(feature=gaze_ps_features)}
            features = {'cameras': tf.train.Feature(bytes_list=tf.train.BytesList(value=camera_features)),
                        'gazemaps': tf.train.Feature(bytes_list=tf.train.BytesList(value=gazemap_features))}
            
            example = tf.train.SequenceExample(
                context=tf.train.Features(features=features),
                feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
            writer.write(example.SerializeToString())
        

