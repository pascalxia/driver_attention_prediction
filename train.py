
import argparse
import sys
import os

import tensorflow as tf 

import networks

import pdb



LEARNING_RATE = 1e-3

def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  camera = features['camera']
  feature_map = features['feature_map']
  gazemap = features['gazemap']
  
  tf.summary.image('camera', camera, max_outputs=4)
  tf.summary.image('gazemap', gazemap, max_outputs=4)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=feature_map)
  
  logits = networks.readout_net(feature_map, (36,64), 0.2)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    accuracy = tf.identity(loss)

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(LEARNING_RATE, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy, name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    #tf.summary.scalar('train_accuracy', accuracy)
    tf.summary.scalar('test_test', accuracy)

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))



# Set up training and evaluation input functions.
def train_input_fn(params):
  """Prepare data for training."""
  
  file_names = [f for f in os.listdir(params.data_folder) if f.endswith('.tfrecords')]
  
  dataset = tf.data.Dataset.from_tensor_slices([params.data_folder+f for f in file_names])
  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  
  def _parse_function(example_proto):
    feature_info = {'camera': tf.FixedLenFeature((), tf.string, default_value=''),
                    'feature_map': tf.FixedLenFeature((), tf.string, default_value=''),
                    'gazemap': tf.FixedLenFeature((), tf.string, default_value='')}
    parsed_features = tf.parse_single_example(example_proto, feature_info)
    
    camera = tf.reshape(tf.decode_raw(parsed_features["camera"], tf.uint8), (720,1280,3))
    feature_map = tf.reshape(tf.decode_raw(parsed_features["feature_map"], tf.float32), (36,64,256))
    gazemap = tf.reshape(tf.decode_raw(parsed_features["gazemap"], tf.uint8), (576,1024,1))
    
    labels = tf.cast(gazemap, tf.float32)
    labels = tf.image.resize_images(labels, (36,64), method=tf.image.ResizeMethod.AREA)
    labels = tf.reshape(labels, (-1,))
    labels = labels/tf.reduce_sum(labels)
    
    features = {}
    features['camera'] = camera
    features['feature_map'] = feature_map
    features['gazemap'] = gazemap
    
    parsed_features["gazemap"] = tf.reshape(tf.decode_raw(parsed_features["gazemap"], tf.uint8), (576,1024))    
    
    return features, labels
    
  dataset = dataset.map(_parse_function)
  
  dataset = dataset.batch(4)
  
  dataset = dataset.repeat()
  
  return dataset


def main(argv):
  
  parser = argparse.ArgumentParser()
  params = parser.parse_args()
  params.data_folder = 'example_data/tfrecords/'
  params.model_dir = 'estimator_logs/'
  
  model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=params.model_dir)
  
  #pdb.set_trace()
  #predict_generator = model.predict(input_fn = lambda: train_input_fn(params))
  #res = next(predict_generator)
  
  # Train and evaluate model.
  model.train(input_fn=lambda: train_input_fn(params))
  
  #pdb.set_trace()
  
  #model.train(input_fn=lambda: train_input_fn(params))

    









if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
