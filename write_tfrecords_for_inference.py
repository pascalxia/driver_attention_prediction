import tensorflow as tf

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import add_args
import random
import feather
from multiprocessing import Pool

import data_point_collector as dpc
import pdb


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

parser = argparse.ArgumentParser()
add_args.for_general(parser)
add_args.for_lstm(parser)
parser.add_argument('--n_divides', type=int, default=1)
parser.add_argument('--n_threads', type=int, default=10)


args = parser.parse_args()

camera_folder = os.path.join(args.data_dir, 'camera_images')
tfrecord_folder = os.path.join(args.data_dir, 'tfrecords')

if not os.path.isdir(tfrecord_folder):
    os.makedirs(tfrecord_folder)

data_point_names = dpc.get_data_point_names(args.data_dir, in_sequences=True,
    longest_seq=args.longest_seq,
    sampleRate=args.sample_rate,
    predictionRate=args.prediction_rate)

splits = [[] for _ in range(args.n_divides)]
for i in range(len(data_point_names)):
    splits[i%args.n_divides].append(data_point_names[i])

def write_one_shard(shard_idx):
    with tf.python_io.TFRecordWriter(
        os.path.join(tfrecord_folder, 
        "cameras_%d.tfrecords" % shard_idx )) as writer:    
        
        for seq in splits[shard_idx]:
            camera_features = list()
            gazemap_features = list()
            gaze_ps_features = list()
            video_id = bytes(seq[0].split('_')[0], encoding = "utf8")
            predicted_time_point_features = list()
            weight_features = list()
            
            for j in range(len(seq) - args.n_future_steps):
                # write camera images
                camera = cv2.imread(os.path.join(camera_folder,seq[j]+'.jpg'))        # do not flip bgr for imencode      
                camera = cv2.resize(
                  camera, 
                  tuple(args.image_size[::-1]),
                  interpolation=cv2.INTER_LINEAR
                )
                camera = cv2.imencode('.jpg', camera)[1].tostring()                     # imencode returns tuple(bool, ndarray)
                camera_features.append(camera)
                
                # write frame names
                time_point = int(seq[j+args.n_future_steps].split('_')[1])
                predicted_time_point_features.append(_int64_feature(time_point))
                
                # write sampling weights
                weight = float(1)
                weight_features.append(tf.train.Feature(float_list=tf.train.FloatList(value=[weight])))
            
            feature_lists = {'predicted_time_points': \
                               tf.train.FeatureList(feature=predicted_time_point_features),
                             'weights': tf.train.FeatureList(feature=weight_features)}
            features = {'cameras': tf.train.Feature(bytes_list=tf.train.BytesList(value=camera_features)),
                        'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id]))}
            
            example = tf.train.SequenceExample(
                context=tf.train.Features(feature=features),
                feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
            writer.write(example.SerializeToString())
    
with Pool(args.n_threads) as pool:
    list(tqdm(pool.imap(write_one_shard, range(args.n_divides)), total=args.n_divides))
            
# for i in tqdm(range(args.n_divides)):
#     write_one_shard(i)

