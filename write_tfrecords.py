import tensorflow as tf

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import random
import h5py


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
            
            # read feature_maps of one video
            video_id = seq[0].split('_')[0]
            feature_maps = h5py.File(os.path.join(feature_folder, video_id+'.h5'), 'r')['X']
            feature_maps = np.transpose(feature_maps, axes=(0,2,3,1))
            
            for j in range(len(seq)):
                # skip the first 4 frames because each feature map is calculated using 4 frame images
                if j < 4:
                    continue
                # when there are no more feature maps for the frame images, break to next video
                if j == len(feature_maps):
                    break
                
                f = seq[j]
                
                camera = cv2.imread(os.path.join(camera_folder,f+'.jpg'))[:,:,[2,1,0]]
                camera = cv2.resize(camera, (64,36), interpolation=cv2.INTER_LINEAR)
                camera_features.append(_bytes_feature(camera.tostring()))
                
                feature_map = feature_maps[j]
                feature_map_features.append(_bytes_feature(feature_map.tostring()))
    
                gazemap = cv2.imread(os.path.join(gazemap_folder, f+'.jpg'))[:,:,0]
                gazemap = cv2.resize(gazemap, (20,12), interpolation=cv2.INTER_AREA)
                gazemap_features.append(_bytes_feature(gazemap.tostring()))
            
            feature_lists = {'cameras': tf.train.FeatureList(feature=camera_features),
                             'feature_maps': tf.train.FeatureList(feature=feature_map_features),
                             'gazemaps': tf.train.FeatureList(feature=gazemap_features)}
                             
            example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
            writer.write(example.SerializeToString())
        

