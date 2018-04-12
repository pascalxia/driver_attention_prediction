import tensorflow as tf

import os
import numpy as np
import cv2


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

feature_folder = 'example_data/application/image_features_alexnet/'
gazemap_folder = 'example_data/application/gazemap_images/'
tfrecord_folder = 'example_data/tfrecords/'



feature_names = [f[:-4] for f in os.listdir(feature_folder) if f.endswith('.npy')]
gazemap_names = [f[:-4] for f in os.listdir(gazemap_folder) if f.endswith('.jpg')]
file_names = list(set(feature_names).intersection(set(gazemap_names)))
file_names.sort()

writer = tf.python_io.TFRecordWriter(tfrecord_folder+file_names[0].split('_')[0]+'.tfrecords')

for f in file_names:
    feature_map = np.load(feature_folder+f+'.npy')
    gazemap = cv2.imread(gazemap_folder+f+'.jpg')[:,:,0]
    feature = {'feature_map': _bytes_feature(feature_map.tostring()),
               'gazemap': _bytes_feature(gazemap.tostring())}
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    writer.write(example.SerializeToString())

writer.close()
