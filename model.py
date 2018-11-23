import tensorflow as tf 
import networks
import numpy as np


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  # input
  cameras = features['cameras']
  camera_input = tf.cast(cameras, tf.float32)
  camera_input = tf.reshape(camera_input, 
                            [-1, params['camera_size'][0], params['camera_size'][1], 3])
  camera_input = camera_input - [123.68, 116.79, 103.939]
  weights = features['weights']
  weights = tf.reshape(weights, (-1,))
  
  video_id = features['video_id']
  predicted_time_points = features['predicted_time_points']
  
  # build up model
  with tf.variable_scope("encoder"):
    readout_network = networks.pure_vgg_encoder(params)
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
    if params['readout'] == 'default':
      readout_net = networks.big_conv_lstm_readout_net
    elif params['readout'] == 'big_conv_lstm':
      readout_net = networks.big_conv_lstm_readout_net
    elif params['readout'] == 'thick_conv_lstm':
      readout_net = networks.thick_conv_lstm_readout_net
    logits, embed, raw_logits = readout_net(feature_map_in_seqs, 
                         feature_map_size=feature_map_size, 
                         drop_rate=0.2,
                         output_embedding=True)
  
  # get prediction
  ps = tf.nn.softmax(logits)
  predictions = {
      'ps': ps
  }
  if 'output_embedding' in params and params['output_embedding']:
    predictions['embed'] = embed
    predictions['raw_logits'] = raw_logits
    predictions['raw_ps'] = tf.nn.softmax(raw_logits)
  predicted_gazemaps = tf.reshape(ps, [-1,]+params['gazemap_size']+[1])
  
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    if 'output_loss' not in params or params['output_loss'] == False:
      predictions['video_id'] = tf.tile(video_id, tf.shape(ps)[0:1])
      predictions['predicted_time_points'] = tf.reshape(predicted_time_points, shape=[-1, 1])
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  
  # set up loss
  labels = tf.reshape(labels, (-1, params['gazemap_size'][0]*params['gazemap_size'][1]))
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
  _, grid_y, grid_x = tf.meshgrid(tf.range(batch_size_tensor*n_steps_tensor),
                                  tf.range(params['gazemap_size'][0]),
                                  tf.range(params['gazemap_size'][1]),
                                  indexing='ij')

  height_trans = params['gazemap_size'][0] * tf_repeat(features['translation'][:, 1], n_steps_tensor)
  width_trans = params['gazemap_size'][1] * tf_repeat(features['translation'][:, 0], n_steps_tensor)
  height_center = params['gazemap_size'][0]/2 + height_trans
  width_center = params['gazemap_size'][1]/2 + width_trans
  height_center = tf.reshape(height_center, [-1, 1, 1])
  width_center = tf.reshape(width_center, [-1, 1, 1])
  dist_matrix = tf.sqrt(tf.square(tf.cast(grid_x, tf.float32)-width_center) + \
                        tf.square(tf.cast(grid_y, tf.float32)-height_center))
  spread = tf.multiply(predicted_gazemaps[..., 0], dist_matrix)
  spread = tf.reduce_sum(spread, axis=[-2, -1])
  spread = tf.metrics.mean(spread)
  
  metrics = {
    'custom_cc': custom_cc,
    'kl': kl,
    'spread': spread,}
    
  if 'output_loss' in params and params['output_loss']:
    predictions['kl'] = kl
    predictions['cc'] = custom_cc
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions['video_id'] = tf.tile(video_id, tf.shape(ps)[0:1])
    predictions['predicted_time_points'] = tf.reshape(predicted_time_points, shape=[-1, 1])
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  # set up summaries
  gazemaps = features['gazemaps']
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
  camera_visual = tf.reshape(cameras, [-1,]+params['camera_size']+[3])
  camera_visual = tf.image.resize_bilinear(camera_visual, params['visual_size'])
  slow_summaries.append(
    tf.summary.image('cameras', camera_visual, max_outputs=2)
  )
  gazemap_visual = tf.reshape(gazemaps, [-1,]+params['gazemap_size']+[1])
  gazemap_visual = tf.image.resize_bilinear(gazemap_visual, params['visual_size'])
  slow_summaries.append(
    tf.summary.image('gazemaps', gazemap_visual, max_outputs=2)
  )
  slow_summaries.append(
    tf.summary.image('predictions', predicted_gazemaps, max_outputs=2)
  )
  for i in range(embed.shape[-1]):
    slow_summaries.append(
      tf.summary.histogram('embed_'+str(i), embed[..., i])
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

def tf_repeat(a_tensor, repeat):
  a_tensor = tf.reshape(a_tensor, [-1, 1])    # Convert to a n x 1 matrix.
  a_tensor = tf.tile(a_tensor, [1, repeat])  # Create multiple columns.
  a_tensor = tf.reshape(a_tensor, [-1])       # Convert back to a vector.
  return a_tensor
  
