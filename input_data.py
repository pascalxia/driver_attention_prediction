import os
import tensorflow as tf 



def get_sample_prob(example):
    sequence_feature_info = {
        'weights': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32)
    }
    _, sequence_features = tf.parse_single_sequence_example(example, 
        sequence_features=sequence_feature_info)
    sample_prob = tf.reduce_mean(sequence_features['weights'])
    return sample_prob


def oversample_classes(example):
    """
    Returns the number of copies of given example
    """
    sample_prob = get_sample_prob(example)
    # for sample_prob smaller than 1, we
    # want to return 1
    sample_prob = tf.maximum(sample_prob, 1) 
    # for low probability classes this number will be very large
    repeat_count = tf.floor(sample_prob)
    # sample_prob can be e.g 1.9 which means that there is still 90%
    # of change that we should return 2 instead of 1
    repeat_residual = sample_prob - repeat_count # a number between 0-1
    residual_acceptance = tf.less_equal(
                        tf.random_uniform([], dtype=tf.float32), repeat_residual
    )
    
    residual_acceptance = tf.cast(residual_acceptance, tf.int64)
    repeat_count = tf.cast(repeat_count, dtype=tf.int64)
    
    return repeat_count + residual_acceptance


def undersampling_filter(example):
    """
    Computes if given example is rejected or not.
    """
    sample_prob = get_sample_prob(example)
    sample_prob = tf.minimum(sample_prob, 1.0)
    
    acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), sample_prob)

    return acceptance


def input_fn(dataset, batch_size, n_steps, shuffle, include_labels, n_epochs, args, weight_data=False):
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
  
  # if apply weighted sampling
  if weight_data:
    dataset = dataset.flat_map(
      lambda x: tf.data.Dataset.from_tensors(x).repeat(oversample_classes(x))
    )
    dataset = dataset.filter(undersampling_filter)
  
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
    
    predicted_time_points = sequence_features["predicted_time_points"]
    weights = sequence_features['weights']
    
    
    if include_labels:
      labels = tf.reshape(tf.decode_raw(sequence_features["gaze_ps"], tf.float32), 
        [-1, args.gazemap_size[0]*args.gazemap_size[1]])
    
    
    def sample_offset():
      """
      sample the starting point (offset) according to the sampling weights of windows
      """
      if weight_data:
        cum_weights = tf.cumsum(weights, axis=0)
        sample_prob = cum_weights[n_steps-1:] - tf.concat([[0,], cum_weights[:-n_steps]], axis=0)
        sample_prob = sample_prob / tf.reduce_sum(sample_prob)
        offset = tf.multinomial(logits=tf.log([sample_prob,]), num_samples=1, output_dtype=tf.int32)[0, 0]
      else:
        offset = tf.random_uniform(shape=[], minval=0, 
                                   maxval=tf.maximum(length-n_steps+1, 1), dtype=tf.int32)
      return offset
    
    if n_steps is not None:
      #select a subsequence
      length = tf.shape(cameras)[0]
      offset = tf.cond(tf.less(length, n_steps), lambda: 0, sample_offset)
          
      end = tf.minimum(offset+n_steps, length)
      cameras = cameras[offset:end]
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
    
    if not weight_data:
      weights = tf.ones(tf.shape(weights))
    else:
      weights = tf.tile(tf.reduce_mean(weights, axis=0, keep_dims=True), [tf.shape(cameras)[0],])
      
    
    # return features and labels
    features = {}
    features['cameras'] = cameras
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
                   'gazemaps': [None,]+args.image_size+[1],
                   'video_id': [],
                   'predicted_time_points': [None,],
                   'weights': [None,]}
                   
  if include_labels:
      padded_shapes = (padded_shapes, [None, args.gazemap_size[0]*args.gazemap_size[1]])
      
  dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
                                                               
  dataset = dataset.prefetch(buffer_size=batch_size)
  
  dataset = dataset.repeat(n_epochs)

  
  return dataset
