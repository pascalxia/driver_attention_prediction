
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
  labels = tf.reshape(labels, (-1, 36*64))
  
  tf.summary.image('camera', tf.reshape(camera, (-1,576,1024,3)), max_outputs=4)
  tf.summary.image('gazemap', tf.reshape(gazemap, (-1,36,64,1)), max_outputs=4)
  
  logits = networks.big_conv_lstm_readout_net(feature_map, 
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
def train_input_fn(params):
  """Prepare data for training."""
  
  file_names = [f for f in os.listdir(params.data_folder) if f.endswith('.tfrecords')]
  camera_gaze_dataset = tf.data.TFRecordDataset(params.data_folder+'camera_gaze.tfrecords')
  image_feature_dataset = tf.data.TFRecordDataset(params.data_folder+'image_features_alexnet.tfrecords')
  dataset = tf.data.Dataset.zip( (camera_gaze_dataset, image_feature_dataset) )
  
  def _parse_function(camera_gaze_example, image_feautre_example):
    feature_info = {'camera': tf.VarLenFeature(dtype=tf.string),
                    'gazemap': tf.VarLenFeature(dtype=tf.string)}
    parsed_features = tf.parse_single_example(camera_gaze_example, feature_info)
    
    feature_info = {'feature_map': tf.VarLenFeature(dtype=tf.string)}
    additional_features = tf.parse_single_example(image_feautre_example, feature_info)
    
    parsed_features.update(additional_features)
    
    for key in parsed_features:
      parsed_features[key] = tf.sparse_tensor_to_dense(parsed_features[key], default_value='')
    
    camera = tf.reshape(tf.decode_raw(parsed_features["camera"], tf.uint8), (-1, 576, 1024, 3))
    feature_map = tf.reshape(tf.decode_raw(parsed_features["feature_map"], tf.float32), (-1,36,64,256))
    gazemap = tf.reshape(tf.decode_raw(parsed_features["gazemap"], tf.uint8), (-1,36,64,1))
    
    labels = tf.cast(gazemap, tf.float32)
    #labels = tf.image.resize_images(labels, (36,64), method=tf.image.ResizeMethod.AREA)
    labels = tf.reshape(labels, (-1,))
    labels = labels/tf.reduce_sum(labels)
    
    features = {}
    features['camera'] = camera
    features['feature_map'] = feature_map
    features['gazemap'] = gazemap
    
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
