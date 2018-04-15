
import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args

import pdb




LEARNING_RATE = 1e-3

def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  cameras = features['cameras']
  feature_maps = features['feature_maps']
  gazemaps = features['gazemaps']
  labels = tf.reshape(labels, (-1, 36*64))
  
  tf.summary.image('cameras', tf.reshape(cameras, (-1,36,64,3)), max_outputs=6)
  tf.summary.image('gazemaps', tf.reshape(gazemaps, (-1,36,64,1)), max_outputs=6)
  
  logits = networks.big_conv_lstm_readout_net(feature_maps, 
                                              feature_map_size=(36,64), 
                                              drop_rate=0.2)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    #TODO: write correlation coefficient as a accuracy metric
    accuracy = tf.identity(loss)

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(LEARNING_RATE, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy, name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy)


    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))



# Set up training and evaluation input functions.
def train_input_fn(args):
  """Prepare data for training."""
  
  dataset = tf.data.TFRecordDataset(os.path.join(args.data_dir,'tfrecords','cameras_gazes_alexnet_features.tfrecords'))
  #image_feature_dataset = tf.data.TFRecordDataset(os.path.join(args.data_dir,'tfrecords','image_features_alexnet.tfrecords'))
  #dataset = tf.data.Dataset.zip( (camera_gaze_dataset, image_feature_dataset) )

  def _parse_function(example):
    # parsing
    feature_info = {'cameras': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string),
                    'feature_maps': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string),
                    'gazemaps': tf.FixedLenSequenceFeature(shape=[], dtype=tf.string)}
    _, parsed_features = tf.parse_single_sequence_example(example, sequence_features=feature_info)
    
    # reshaping
    cameras = tf.reshape(tf.decode_raw(parsed_features["cameras"], tf.uint8), (-1, 36, 64, 3))
    feature_maps = tf.reshape(tf.decode_raw(parsed_features["feature_maps"], tf.float32), (-1,36,64,256))
    gazemaps = tf.reshape(tf.decode_raw(parsed_features["gazemaps"], tf.uint8), (-1,36,64,1))
    
    #select a subsequence
    length = tf.shape(cameras)[0]
    #pdb.set_trace()
    offset = tf.random_uniform(shape=[], minval=0, maxval=tf.maximum(length-args.n_steps+1, 1), dtype=tf.int32)
    end = tf.minimum(offset+args.n_steps, length)
    cameras = cameras[offset:end]
    feature_maps = feature_maps[offset:end]
    gazemaps = gazemaps[offset:end]
    
    # normalizing gazemap into probability distribution
    labels = tf.cast(gazemaps, tf.float32)
    #labels = tf.image.resize_images(labels, (36,64), method=tf.image.ResizeMethod.AREA)
    labels = tf.reshape(labels, (-1, 36*64))
    labels = tf.matmul(tf.diag(1/tf.reduce_sum(labels,axis=1)), labels)
    #labels = labels/tf.reduce_sum(labels, axis=1)

    
    # return features and labels
    features = {}
    features['cameras'] = cameras
    features['feature_maps'] = feature_maps
    features['gazemaps'] = gazemaps
    
    return features, labels
  
  dataset = dataset.map(_parse_function)
  
  dataset = dataset.padded_batch(args.batch_size, padded_shapes=({'cameras': [None,36, 64, 3],
                                                                  'feature_maps': [None,36,64,256],
                                                                  'gazemaps': [None,36,64,1]},
                                                                 [None,36*64]))
  
  dataset = dataset.repeat()
  
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
  
  # Train and evaluate model.
  model.train(input_fn=lambda: train_input_fn(args))
  
  #pdb.set_trace()
  
  #model.train(input_fn=lambda: train_input_fn(args))

    









if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
