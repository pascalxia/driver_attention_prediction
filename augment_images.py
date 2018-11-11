import tensorflow as tf
import pdb

MAX_LIGHTNESS_DELTA = 0.15
CONTRAST_FACTOR_RANGE = [0.8, 1.2]
MAX_TRANSLATION = 0.15




def augment_images(cameras, gazemaps=None):
  # convert to float
  cameras = tf.image.convert_image_dtype(cameras, tf.float32)
  
  # alter lightness
  cameras = tf.image.random_brightness(cameras, MAX_LIGHTNESS_DELTA)
  
  # alter contrast
  cameras = tf.image.random_contrast(
    cameras, 
    CONTRAST_FACTOR_RANGE[0],
    CONTRAST_FACTOR_RANGE[1])
  
  # handle range issues  
  cameras = tf.minimum(cameras, 1.0)
  cameras = tf.maximum(cameras, 0.0)
  
  # convert back to uint8
  cameras = tf.image.convert_image_dtype(cameras, tf.uint8)
  
  # translation
  relative_transloations = tf.random_uniform(
    [2,],
    minval=-MAX_TRANSLATION,
    maxval=MAX_TRANSLATION)
  translations = tf.multiply(
    relative_transloations, tf.cast(tf.shape(cameras)[1:3][::-1], tf.float32))
  #pdb.set_trace()
  translations = tf.round(translations)
  cameras = tf.contrib.image.translate(
    cameras,
    translations)
    
  if gazemaps is not None:
    translations = tf.multiply(
      relative_transloations, tf.cast(tf.shape(gazemaps)[1:3][::-1], tf.float32))
    translations = tf.round(translations)
    gazemaps = tf.contrib.image.translate(
      gazemaps,
      translations)
    return cameras, gazemaps, relative_transloations
  else:
    return cameras, relative_transloations
  
