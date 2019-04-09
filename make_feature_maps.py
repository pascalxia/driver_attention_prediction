import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args
from keras import backend as K
import shutil
import numpy as np
import scipy.misc as misc



def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    # input
    cameras = features['cameras']  
    video_id = features['video_id']
    predicted_time_points = features['predicted_time_points']
  
    # build up model
    #set up encoder net-----------------
    input_tensor = tf.reshape(tf.cast(cameras, tf.float32),  
        [-1, params['image_size'][0], params['image_size'][1], 3])
    input_tensor = input_tensor - [123.68, 116.79, 103.939]
  
    with tf.variable_scope("encoder"):
        feature_net = networks.alex_encoder(params)
        feature_maps = feature_net(input_tensor)
    
    predictions = {
        'feature_maps': feature_maps
    }
  
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions['video_id'] = tf.tile(video_id, tf.shape(feature_maps)[0:1])
        predictions['predicted_time_points'] = tf.reshape(predicted_time_points, shape=[-1, 1])
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def input_fn(dataset, batch_size, n_steps, shuffle, n_epochs, args):
    """Prepare data for training."""
  
    # get and shuffle tfrecords files
    
    if dataset is not None:
        file_path = os.path.join(args.data_dir, dataset, 'tfrecords', 'cameras_*.tfrecords')
    else:
        file_path = os.path.join(args.data_dir, 'tfrecords', 'cameras_*.tfrecords')
    files = tf.data.Dataset.list_files(file_path)
    
    if shuffle:
        files = files.shuffle(buffer_size=10)
  
    # parellel interleave to get raw bytes
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=5, block_length=batch_size))
  
    # shuffle before parsing
    if shuffle:
        dataset = dataset.shuffle(buffer_size=5*batch_size)
  
    # parse data
    def _parse_function(example):
        # parsing
        context_feature_info = {
            'video_id': tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
        sequence_feature_info = {
            'cameras': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string),
            'predicted_time_points': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }
        context_features, sequence_features = tf.parse_single_sequence_example(example, 
            context_features=context_feature_info,
            sequence_features=sequence_feature_info)
    
        video_id = context_features['video_id']
        cameras = tf.reshape(tf.decode_raw(sequence_features["cameras"], tf.uint8), 
            [-1,]+args.image_size+[3])
        predicted_time_points = sequence_features["predicted_time_points"]
    
        if n_steps is not None:
            #select a subsequence
            length = tf.shape(cameras)[0]

            offset = tf.random_uniform(shape=[], minval=0, maxval=tf.maximum(length-n_steps+1, 1), dtype=tf.int32)
            end = tf.minimum(offset+n_steps, length)
            cameras = cameras[offset:end]
            feature_maps = feature_maps[offset:end]
            gazemaps = gazemaps[offset:end]
            predicted_time_points = predicted_time_points[offset:end]
    
        # return features and labels
        features = {}
        features['cameras'] = cameras
        features['video_id'] = video_id
        features['predicted_time_points'] = predicted_time_points

        return features
  
    dataset = dataset.map(_parse_function, num_parallel_calls=10)
  
    padded_shapes = {'cameras': [None,]+args.image_size+[3],
                     'video_id': [],
                     'predicted_time_points': [None,]}
      
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
                                                               
    dataset = dataset.prefetch(buffer_size=batch_size)
  
    dataset = dataset.repeat(n_epochs)

  
    return dataset




def main(argv):
    parser = argparse.ArgumentParser()
    add_args.for_general(parser)
    add_args.for_inference(parser)
    add_args.for_evaluation(parser)
    add_args.for_feature(parser)
    add_args.for_lstm(parser)
    args = parser.parse_args()
    
    config = tf.estimator.RunConfig(save_summary_steps=float('inf'),
                                      log_step_count_steps=10)
    
    params = {
        'image_size': args.image_size,
        'gazemap_size': args.gazemap_size,
        'model_dir': args.model_dir
      }
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        config=config,
        params=params)
    
    #determine which checkpoint to restore
    if args.model_iteration is None:
        best_ckpt_dir = os.path.join(args.model_dir, 'best_ckpt')
        if os.path.isdir(best_ckpt_dir):
            ckpt_name = [f.split('.index')[0] for f in os.listdir(best_ckpt_dir) if f.endswith('.index')][0]
            ckpt_path = os.path.join(best_ckpt_dir, ckpt_name)
            args.model_iteration = ckpt_name.split('-')[1]
    else:
        ckpt_name = 'model.ckpt-'+args.model_iteration
        ckpt_path = os.path.join(args.model_dir, ckpt_name)    
    

    K.clear_session()
    predict_generator = model.predict(
        input_fn = lambda: input_fn(None, 
            batch_size=1, n_steps=None, 
            shuffle=False,
            n_epochs=1, args=args),
        checkpoint_path=ckpt_path)
    
    output_dir = os.path.join(args.data_dir, 'image_features_'+args.feature_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    previous_video_id = None
    for res in predict_generator:
        if previous_video_id is None:
            print('Start inference for video: %s' % res['video_id'])
            previous_video_id = res['video_id']
        elif res['video_id'] != previous_video_id:
            print('Start inference for video: %s' % res['video_id'])
            previous_video_id = res['video_id']
            
        output_path = os.path.join(output_dir, 
            str(res['video_id'])+'_'+str(res['predicted_time_points'][0]).zfill(5)+'.npy')
        
        feature_map = res['feature_maps']
        np.save(output_path, feature_map)




if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
