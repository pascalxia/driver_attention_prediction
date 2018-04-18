
import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args
from keras import backend as K

import pdb




LEARNING_RATE = 1e-3

def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  # input
  cameras = features['cameras']
  feature_maps = features['feature_maps']
  gazemaps = features['gazemaps']
  labels = tf.reshape(labels, (-1, 12*20))
  
  tf.summary.image('cameras', tf.reshape(cameras, (-1,36,64,3)), max_outputs=2)
  tf.summary.image('gazemaps', tf.reshape(gazemaps, (-1,12,20,1)), max_outputs=2)
  
  # build up model
  logits = networks.big_conv_lstm_readout_net(feature_maps, 
                                              feature_map_size=(12,20), 
                                              drop_rate=0.2)
  
  # get prediction
  ps = tf.nn.softmax(logits)
  predictions = {
      'ps': ps
  }
  
  predicted_gazemaps = tf.reshape(ps, (-1, 12, 20, 1))
  tf.summary.image('predictions', predicted_gazemaps, max_outputs=2)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  # set up loss
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
  
  # set up training
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
  else:
    train_op = None
    
    
  # set up metrics
  #TODO: write correlation coefficient as a accuracy metric
  accuracy = tf.contrib.metrics.streaming_pearson_correlation(ps, labels)
  metrics = {'accuracy': accuracy}
  
  tf.summary.scalar('train_accuracy', accuracy[1])
    
  
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics)
    
  



# Set up training and evaluation input functions.
def input_fn(dataset, batch_size, n_steps, shuffle, include_labels, n_epochs, args):
  """Prepare data for training."""
  
  # get and shuffle tfrecords files
  files = tf.data.Dataset.list_files(os.path.join(args.data_dir, dataset,'tfrecords','cameras_gazes_'+args.feature_name+'_features_*.tfrecords'))
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
    feature_info = {'cameras': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string),
                    'feature_maps': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string),
                    'gazemaps': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string)}
    _, parsed_features = tf.parse_single_sequence_example(example, sequence_features=feature_info)
    
    # reshaping
    cameras = tf.reshape(tf.decode_raw(parsed_features["cameras"], tf.uint8), (-1, 36, 64, 3))
    feature_maps = tf.reshape(tf.decode_raw(parsed_features["feature_maps"], tf.float32), (-1,12,20,64))
    gazemaps = tf.reshape(tf.decode_raw(parsed_features["gazemaps"], tf.uint8), (-1,12,20,1))
    
    if n_steps is not None:
        #select a subsequence
        length = tf.shape(cameras)[0]
        #pdb.set_trace()
        offset = tf.random_uniform(shape=[], minval=0, maxval=tf.maximum(length-n_steps+1, 1), dtype=tf.int32)
        end = tf.minimum(offset+n_steps, length)
        cameras = cameras[offset:end]
        feature_maps = feature_maps[offset:end]
        gazemaps = gazemaps[offset:end]
    
    if include_labels:
        # normalizing gazemap into probability distribution
        labels = tf.cast(gazemaps, tf.float32)
        #labels = tf.image.resize_images(labels, (36,64), method=tf.image.ResizeMethod.AREA)
        labels = tf.reshape(labels, (-1, 12*20))
        labels = tf.matmul(tf.diag(1/tf.reduce_sum(labels,axis=1)), labels)

    
    # return features and labels
    features = {}
    features['cameras'] = cameras
    features['feature_maps'] = feature_maps
    features['gazemaps'] = gazemaps
    
    if include_labels:
        return features, labels
    else:
        return features

  dataset = dataset.map(_parse_function, num_parallel_calls=10)
  
  padded_shapes = {'cameras': [None,36, 64, 3],
                   'feature_maps': [None,12,20,64],
                   'gazemaps': [None,12,20,1]}
  if include_labels:
      padded_shapes = (padded_shapes, [None,12*20])
      
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
  ds = train_input_fn(args)
  iterator = ds.make_one_shot_iterator()
  next_element = iterator.get_next()
  sess = tf.Session()
  pdb.set_trace()
  res = sess.run(next_element)
  '''
  
  config = tf.estimator.RunConfig(save_summary_steps=10,
                                  log_step_count_steps=10)
  
  model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    config=config)
  
  #pdb.set_trace()
  #predict_generator = model.predict(input_fn = lambda: train_input_fn(args))
  #res = next(predict_generator)
  
  for _ in range(args.train_epochs // args.epochs_before_validation):
    # Train the model.
    #pdb.set_trace()
    K.clear_session()
    model.train(input_fn=lambda: input_fn('training',
      args.batch_size, args.n_steps, 
      shuffle=True, include_labels=True, 
      n_epochs=args.epochs_before_validation, args=args) )
    # validate the model
    K.clear_session()
    valid_results = model.evaluate(input_fn=lambda: input_fn('validation', 
      batch_size=1, n_steps=None, 
      shuffle=False, include_labels=True, 
      n_epochs=1, args=args) )
    print(valid_results)
  
  #pdb.set_trace()
  
  #model.train(input_fn=lambda: train_input_fn(args))

    









if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
