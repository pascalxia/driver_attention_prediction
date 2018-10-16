import tensorflow as tf 
import networks
import numpy as np


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  # input
  cameras = features['cameras']
  camera_input = tf.cast(cameras, tf.float32)
  camera_input = tf.reshape(camera_input, 
                            [-1, params['image_size'][0], params['image_size'][1], 3])
  camera_input = camera_input - [123.68, 116.79, 103.939]
  gazemaps = features['gazemaps']
  

  weights = features['weights']
  weights = tf.reshape(weights, (-1,))

  labels = tf.reshape(labels, (-1, params['gazemap_size'][0]*params['gazemap_size'][1]))
  
  video_id = features['video_id']
  predicted_time_points = features['predicted_time_points']
  
  # build up model
  with tf.variable_scope("encoder"):
    readout_network = networks.alex_encoder(params)
    feature_maps = readout_network(camera_input)
    batch_size_tensor = tf.shape(cameras)[0]
    n_steps_tensor = tf.shape(cameras)[1]
    feature_map_size = (int(feature_maps.get_shape()[1]), 
                        int(feature_maps.get_shape()[2]))
    n_channel = int(feature_maps.get_shape()[3])
    feature_map_in_seqs = tf.reshape(feature_maps,
                                     [batch_size_tensor, n_steps_tensor,
                                      feature_map_size[0], feature_map_size[1],
                                      n_channel])
  with tf.variable_scope("readout"):
    logits = networks.thick_conv_lstm_readout_net(feature_map_in_seqs, 
                                                  feature_map_size=params['feature_map_size'], 
                                                  drop_rate=0.2)
  
  # get prediction
  ps = tf.nn.softmax(logits)
  predictions = {
      'ps': ps
  }
  predicted_gazemaps = tf.reshape(ps, [-1,]+params['gazemap_size']+[1])
  
  # set up loss
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
  
  # set up training
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           scope='readout')
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step(),
                                  var_list=variables_to_train)
  else:
    train_op = None
    
  # set up metrics
  # Calculate correlation coefficient
  s1 = ps - tf.reduce_mean(ps, axis=1, keepdims=True)
  s2 = labels - tf.reduce_mean(labels, axis=1, keepdims=True)
  custom_cc = tf.reduce_sum(tf.multiply(s1, s2), axis=1)/tf.sqrt(tf.reduce_sum(tf.pow(s1,2), axis=1)*tf.reduce_sum(tf.pow(s2,2), axis=1))
  #custom_cc = weights*custom_cc
  # Exclude NaNs.
  mask = tf.is_finite(custom_cc)
  custom_cc = tf.boolean_mask(custom_cc, mask)
  custom_cc = tf.metrics.mean(custom_cc)
  
  # Calculate KL-divergence
  _labels = tf.maximum(labels, params['epsilon'])
  p_entropies = tf.reduce_sum(-tf.multiply(_labels, tf.log(_labels)), axis=1)
  kls = loss - p_entropies
  #kls = weights*kls
  kl = tf.metrics.mean(kls)
    
  # Calculate spreadness
  grid_x, grid_y = np.meshgrid(np.arange(params['gazemap_size'][1]), 
                               np.arange(params['gazemap_size'][0]))
  dist_matrix = np.sqrt(np.square(grid_x-params['gazemap_size'][1]/2) + \
                        np.square(grid_y-params['gazemap_size'][0]/2))
  spread = tf.multiply(ps, np.reshape(dist_matrix, (-1,)))
  spread = tf.reduce_sum(spread, axis=-1)
  spread = tf.metrics.mean(spread)
  
  metrics = {
    'custom_cc': custom_cc,
    'kl': kl,
    'spread': spread,}
  
  # set up summaries
  quick_summaries = []
  quick_summaries.append(tf.summary.scalar('kl', kl[1]))
  quick_summaries.append(tf.summary.scalar('custom_cc', custom_cc[1]))
  quick_summaries.append(tf.summary.scalar('spread', spread[1]))
  quick_summaries.append(tf.summary.scalar('loss', loss))
  quick_summaries.append(tf.summary.scalar('mean_weight', tf.reduce_mean(weights)))
  quick_summary_op = tf.summary.merge(quick_summaries, name='quick_summary')
  quick_summary_hook = tf.train.SummarySaverHook(
    params['quick_summary_period'],
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
    params['slow_summary_period'],
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
