import tensorflow as tf

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import add_args
import random
import pdb


import data_point_collector as dpc


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    


parser = argparse.ArgumentParser()
add_args.for_general(parser)
add_args.for_lstm(parser)
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
##################### DEBUG ######################
data_point_names = data_point_names[:20]

random.shuffle(data_point_names)
splits = [[] for _ in range(args.n_divides)]
for i in range(len(data_point_names)):
    splits[i%args.n_divides].append(data_point_names[i])

for i in range(len(splits)):
    with tf.python_io.TFRecordWriter(
        os.path.join(tfrecord_folder, 
        "cameras_gazes_%s_features_%dfuture_%d.tfrecords" \
          % (args.feature_name, args.n_future_steps, i) )) as writer:
        
        for seq in tqdm(splits[i]):
            camera_features = list()
            feature_map_features = list()
            gazemap_features = list()
            gaze_ps_features = list()
            video_id = int(seq[0].split('_')[0])
            predicted_time_point_features = list()
            for j in range(len(seq) - args.n_future_steps):
                # write camera images
                camera = cv2.imread(os.path.join(camera_folder,seq[j]+'.jpg'))               # do not flip bgr for imencode
                camera = cv2.resize(
                  camera, 
                  tuple(args.image_size[::-1]),
                  interpolation=cv2.INTER_LINEAR
                ) # please check if this is the desired size
                camera = cv2.imencode('.jpg', camera)[1].tostring()                     # imencode returns tuple(bool, ndarray)
                camera_features.append(camera)
                
                # write image feature maps
                feature_map = np.load(os.path.join(feature_folder, seq[j]+'.npy'))
                feature_map_features.append(_bytes_feature(feature_map.tostring()))
                
                # write gaze probability distribution
                #pdb.set_trace()
                gazemap = cv2.imread(os.path.join(gazemap_folder, 
                  seq[j+args.n_future_steps]+'.jpg'))[:,:,0]
                gaze_ps = gazemap.astype(np.float32)
                gaze_ps = cv2.resize(
                  gaze_ps, 
                  tuple(args.gazemap_size[::-1]), 
                  interpolation=cv2.INTER_AREA
                )
                gaze_ps = gaze_ps.reshape((args.gazemap_size[0]*args.gazemap_size[1],))
                gaze_ps = gaze_ps/np.sum(gaze_ps)
                gaze_ps_features.append(_bytes_feature(gaze_ps.tostring()))
                
                # write gazemap images
                gazemap = cv2.resize(
                  gazemap, 
                  tuple(args.image_size[::-1]), 
                  interpolation=cv2.INTER_AREA
                ) # please check this size as well
                gazemap = cv2.imencode('.jpg', gazemap)[1].tostring()
                gazemap_features.append(gazemap)
                
                # write frame names
                time_point = int(seq[j+args.n_future_steps].split('_')[1])
                predicted_time_point_features.append(_int64_feature(time_point))
            
            feature_lists = {'feature_maps': tf.train.FeatureList(feature=feature_map_features),
                             'gaze_ps': tf.train.FeatureList(feature=gaze_ps_features),
                             'predicted_time_points': \
                               tf.train.FeatureList(feature=predicted_time_point_features)}
            features = {'cameras': tf.train.Feature(bytes_list=tf.train.BytesList(value=camera_features)),
                        'gazemaps': tf.train.Feature(bytes_list=tf.train.BytesList(value=gazemap_features)),
                        'video_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[video_id]))}
            
            example = tf.train.SequenceExample(
                context=tf.train.Features(feature=features),
                feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
            writer.write(example.SerializeToString())
        

