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
    
    if not weight_data:
      weights = tf.ones(tf.shape(weights))
    else:
      weights = tf.tile(tf.reduce_mean(weights, axis=0, keep_dims=True), [tf.shape(cameras)[0],])
      
    
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
                   
  if include_labels:
      padded_shapes = (padded_shapes, [None, args.gazemap_size[0]*args.gazemap_size[1]])
      
  dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
                                                               
  dataset = dataset.prefetch(buffer_size=batch_size)
  
  dataset = dataset.repeat(n_epochs)

  
  return dataset