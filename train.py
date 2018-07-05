
import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args
from keras import backend as K
import shutil

import pdb




LEARNING_RATE = 1e-3



def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  # input
  cameras = features['cameras']
  feature_maps = features['feature_maps']
  gazemaps = features['gazemaps']
  if params['weight_data']:
    weights = features['weights']
    weights = tf.reshape(weights, (-1,))
  else:
    weights = 1.0
  labels = tf.reshape(labels, (-1, params['gazemap_size'][0]*params['gazemap_size'][1]))
  
  video_id = features['video_id']
  predicted_time_points = features['predicted_time_points']
  
  # build up model
  logits = networks.big_conv_lstm_readout_net(feature_maps, 
                                              feature_map_size=params['feature_map_size'], 
                                              drop_rate=0.2)
  
  # get prediction
  ps = tf.nn.softmax(logits)
  predictions = {
      'ps': ps
  }
  predicted_gazemaps = tf.reshape(ps, [-1,]+params['gazemap_size']+[1])
  
  # set up loss
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=weights)
  
  # set up training
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
  else:
    train_op = None
    
  # set up metrics
  # Calculate correlation coefficient
  s1 = ps - tf.reduce_mean(ps, axis=1, keepdims=True)
  s2 = labels - tf.reduce_mean(labels, axis=1, keepdims=True)
  custom_cc = tf.reduce_sum(tf.multiply(s1, s2), axis=1)/tf.sqrt(tf.reduce_sum(tf.pow(s1,2), axis=1)*tf.reduce_sum(tf.pow(s2,2), axis=1))
  custom_cc = weights*custom_cc
  # Exclude NaNs.
  mask = tf.logical_not(tf.is_finite(custom_cc))
  custom_cc = tf.boolean_mask(custom_cc, mask)
  custom_cc = tf.metrics.mean(custom_cc)
  
  # Calculate KL-divergence
  _labels = tf.maximum(labels, params['epsilon'])
  p_entropies = tf.reduce_sum(-tf.multiply(_labels, tf.log(_labels)), axis=1)
  kls = loss - p_entropies
  kls = weights*kls
  kl = tf.metrics.mean(kls)
  
  metrics = {
    'custom_cc': custom_cc,
    'kl': kl,}
  
  
  # set up summaries
  quick_summaries = []
  quick_summaries.append(tf.summary.scalar('accuracy', accuracy[1]))
  quick_summaries.append(tf.summary.scalar('custom_cc', custom_cc[1]))
  quick_summaries.append(tf.summary.scalar('loss', loss))
  quick_summary_op = tf.summary.merge(quick_summaries, name='quick_summary')
  quick_summary_hook = tf.train.SummarySaverHook(
    10,
    output_dir=params['model_dir'],
    summary_op=quick_summary_op
  )
    
  # slow summary
  slow_summaries = []
  slow_summaries.append(
    tf.summary.image('cameras', tf.reshape(cameras, [-1,]+params['image_size']+[3]), max_outputs=2)
  )
  slow_summaries.append(
    tf.summary.image('gazemaps', tf.reshape(gazemaps, [-1,]+params['image_size']+[1]), max_outputs=2)
  )
  slow_summaries.append(
    tf.summary.image('predictions', predicted_gazemaps, max_outputs=2)
  )
  slow_summary_op = tf.summary.merge(slow_summaries, name='slow_summary')
  slow_summary_hook = tf.train.SummarySaverHook(
    50,
    output_dir=params['model_dir'],
    summary_op=slow_summary_op
  )
    
  
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics,
    training_hooks=[quick_summary_hook, slow_summary_hook])
    
  



# Set up training and evaluation input functions.
def input_fn(dataset, batch_size, n_steps, shuffle, include_labels, n_epochs, args):
  """Prepare data for training."""
  
  # get and shuffle tfrecords files
  files = tf.data.Dataset.list_files(os.path.join(args.data_dir, dataset, 'tfrecords_weighted',
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
      'predicted_time_points': tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
      'weights': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32)
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
    weights = sequence_features['weights']
    
    
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
      weights = weights[offset:end]
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
    features['weights'] = weights
    
    if include_labels:
        return features, labels
    else:
        return features
  
  dataset = dataset.map(_parse_function, num_parallel_calls=10)
  
  padded_shapes = {'cameras': [None,]+args.image_size+[3],
                   'feature_maps': [None,]+args.feature_map_size+[args.feature_map_channels],
                   'gazemaps': [None,]+args.image_size+[1],
                   'video_id': [],
                   'predicted_time_points': [None,],
                   'weights': [None,]}
                   
  #padded_shapes = {'feature_maps': [None,]+args.feature_map_size+[args.feature_map_channels]}
                   
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
  add_args.for_feature(parser)
  add_args.for_training(parser)
  add_args.for_lstm(parser)
  args = parser.parse_args()
  
  '''
  this_input_fn=lambda: input_fn('training',
      args.batch_size, args.n_steps, 
      shuffle=True, include_labels=True, 
      n_epochs=args.epochs_before_validation, args=args)
  ds = this_input_fn()
  iterator = ds.make_one_shot_iterator()
  next_element = iterator.get_next()
  sess = tf.Session()
  pdb.set_trace()
  res = sess.run(next_element)
  '''
  
  config = tf.estimator.RunConfig(save_summary_steps=float('inf'),
                                  log_step_count_steps=10)
                                  
  params = {
    'image_size': args.image_size,
    'gazemap_size': args.gazemap_size,
    'feature_map_size': args.feature_map_size,
    'model_dir': args.model_dir,
    'weight_data': args.weight_data
  }
  
  model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    config=config,
    params=params)
  
  #pdb.set_trace()
  #predict_generator = model.predict(input_fn = lambda: train_input_fn(args))
  #res = next(predict_generator)
  
  # set up the directory to save the best checkpoint
  best_ckpt_dir = os.path.join(args.model_dir, 'best_ckpt')
  if not os.path.isdir(best_ckpt_dir) or len(os.listdir(best_ckpt_dir))==0:
    smallest_loss = float('Inf')
    if not os.path.isdir(best_ckpt_dir):
      os.makedirs(best_ckpt_dir)
  else:
    smallest_loss = [float(f.split('_')[1]) for f in os.listdir(best_ckpt_dir) if f.startswith('loss_')][0]
  
  for _ in range(args.train_epochs // args.epochs_before_validation):
    # Train the model.
    K.clear_session()
    model.train(input_fn=lambda: input_fn('training',
      args.batch_size, args.n_steps, 
      shuffle=True, include_labels=True, 
      n_epochs=args.epochs_before_validation, args=args)
    )
    # validate the model
    K.clear_session()
    valid_results = model.evaluate(input_fn=lambda: input_fn('validation', 
      batch_size=1, n_steps=None, 
      shuffle=False, include_labels=True, 
      n_epochs=1, args=args) )
    print(valid_results)
    
    if valid_results['loss'] < smallest_loss:
      smallest_loss = valid_results['loss']
      # delete best_ckpt_dir
      shutil.rmtree(best_ckpt_dir)
      # re-make best_ckpt_dir as empty
      os.makedirs(best_ckpt_dir)
      # note down the new smallest loss
      open(os.path.join(best_ckpt_dir, 'loss_%f' % smallest_loss), 'a').close()
      # copy the checkpoint
      files_to_copy = [f for f in os.listdir(args.model_dir) 
        if f.startswith('model.ckpt-'+str(valid_results['global_step']))]
      for f in files_to_copy:
        shutil.copyfile(os.path.join(args.model_dir, f),
          os.path.join(best_ckpt_dir, f))
  
  
  
  #model.train(input_fn=lambda: train_input_fn(args))

    









if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
