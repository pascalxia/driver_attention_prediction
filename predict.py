
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

import pdb



def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  # input
  cameras = features['cameras']
  feature_maps = features['feature_maps']
  gazemaps = features['gazemaps']
  
  video_id = features['video_id']
  predicted_time_points = features['predicted_time_points']
  
  # build up model
  logits = networks.thick_conv_lstm_readout_net(feature_maps, 
                                              feature_map_size=params['feature_map_size'], 
                                              drop_rate=0.2)
  
  # get prediction
  ps = tf.nn.softmax(logits)
  predictions = {
      'ps': ps
  }
  predicted_gazemaps = tf.reshape(ps, [-1,]+params['gazemap_size']+[1])
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions['video_id'] = tf.tile(video_id, tf.shape(ps)[0:1])
    predictions['predicted_time_points'] = tf.reshape(predicted_time_points, shape=[-1, 1])
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  


# Set up input functions.
def input_fn(dataset, batch_size, n_steps, shuffle, include_labels, n_epochs, args):
  """Prepare data for training."""
  
  # get and shuffle tfrecords files
  files = tf.data.Dataset.list_files(os.path.join(args.data_dir, dataset, 'tfrecords',
    'cameras_gazes_'+args.feature_name+\
    '_features_%dfuture_*.tfrecords' % args.n_future_steps))
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
      'cameras': tf.VarLenFeature(dtype=tf.string),
      'gazemaps': tf.VarLenFeature(dtype=tf.string),
      'video_id': tf.FixedLenFeature(shape=[], dtype=tf.int64)
    }
    sequence_feature_info = {
      'feature_maps': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string),
      'gaze_ps': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string),
      'predicted_time_points': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
    }
    context_features, sequence_features = tf.parse_single_sequence_example(example, 
      context_features=context_feature_info,
      sequence_features=sequence_feature_info)
    
    cameras = tf.sparse_tensor_to_dense(context_features["cameras"], default_value='')
    gazemaps = tf.sparse_tensor_to_dense(context_features["gazemaps"], default_value='')
    video_id = context_features['video_id']
    
    feature_maps = tf.reshape(tf.decode_raw(sequence_features["feature_maps"], tf.float32), 
      [-1,]+args.feature_map_size+[args.feature_map_channels])
    predicted_time_points = sequence_features["predicted_time_points"]
    
    
    if include_labels:
      labels = tf.reshape(tf.decode_raw(sequence_features["gaze_ps"], tf.float32), 
        [-1, args.gazemap_size[0]*args.gazemap_size[1]])
    
    
    if n_steps is not None:
      #select a subsequence
      length = tf.shape(cameras)[0]
      
      offset = tf.random_uniform(shape=[], minval=0, maxval=tf.maximum(length-n_steps+1, 1), dtype=tf.int32)
      end = tf.minimum(offset+n_steps, length)
      cameras = cameras[offset:end]
      feature_maps = feature_maps[offset:end]
      gazemaps = gazemaps[offset:end]
      predicted_time_points = predicted_time_points[offset:end]
      if include_labels:
        labels = labels[offset:end]
    
    # decode jpg's
    cameras = tf.map_fn(
      tf.image.decode_jpeg,
      cameras,
      dtype=tf.uint8,
      back_prop=False
    )
    gazemaps = tf.map_fn(
      tf.image.decode_jpeg,
      gazemaps,
      dtype=tf.uint8,
      back_prop=False
    )
    
    
    # return features and labels
    features = {}
    features['cameras'] = cameras
    features['feature_maps'] = feature_maps
    features['gazemaps'] = gazemaps
    features['video_id'] = video_id
    features['predicted_time_points'] = predicted_time_points
    
    if include_labels:
        return features, labels
    else:
        return features
  
  dataset = dataset.map(_parse_function, num_parallel_calls=10)
  
  padded_shapes = {'cameras': [None,]+args.image_size+[3],
                   'feature_maps': [None,]+args.feature_map_size+[args.feature_map_channels],
                   'gazemaps': [None,]+args.image_size+[1],
                   'video_id': [],
                   'predicted_time_points': [None,]}
                   
  if include_labels:
      padded_shapes = (padded_shapes, [None, args.gazemap_size[0]*args.gazemap_size[1]])
      
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
    'feature_map_size': args.feature_map_size,
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
    ckpt_name = 'model.ckpt-'+model_iteration
    ckpt_path = os.path.join(args.model_dir, ckpt_name)
  
  predict_generator = model.predict(
    input_fn = lambda: input_fn('validation', 
      batch_size=1, n_steps=None, 
      shuffle=False, include_labels=False, 
      n_epochs=1, args=args),
    checkpoint_path=ckpt_path)
    
  
  output_dir = os.path.join(args.model_dir, 'prediction_iter_'+args.model_iteration)
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  for res in predict_generator:
    output_path = os.path.join(output_dir, 
      str(res['video_id'])+'_'+str(res['predicted_time_points'][0]).zfill(5)+'.jpg')
    gazemap = np.reshape(res['ps'], args.gazemap_size)
    misc.imsave(output_path, gazemap)



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
