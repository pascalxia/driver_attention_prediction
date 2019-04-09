import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args
from keras import backend as K
import shutil
import numpy as np
import scipy.misc as misc

from input_data import input_fn
from model import model_fn

import pdb

def main(argv):
    parser = argparse.ArgumentParser()
    add_args.for_general(parser)
    add_args.for_inference(parser)
    add_args.for_evaluation(parser)
    add_args.for_feature(parser)
    add_args.for_lstm(parser)
    args = parser.parse_args()
    
    config = tf.estimator.RunConfig(save_summary_steps=float('inf'),
                                      log_step_count_steps=10)
    
    params = {
        'camera_size': args.camera_size,
        'gazemap_size': args.gazemap_size,
        'model_dir': args.model_dir,
        'readout': args.readout,
      }
      
    if args.visible_gpus is not None:
      os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        config=config,
        params=params)
    
    #determine which checkpoint to restore
    if args.model_iteration is None:
        best_ckpt_dir = os.path.join(args.model_dir, 'best_ckpt')
        if os.path.isdir(best_ckpt_dir):
            ckpt_name = [f.split('.index')[0] for f in os.listdir(best_ckpt_dir) if f.endswith('.index')][0]
            ckpt_path = os.path.join(best_ckpt_dir, ckpt_name)
            args.model_iteration = ckpt_name.split('-')[1]
    else:
        ckpt_name = 'model.ckpt-'+args.model_iteration
        ckpt_path = os.path.join(args.model_dir, ckpt_name)
    
    predict_generator = model.predict(
        input_fn = lambda: input_fn('inference', 
            batch_size=1, n_steps=None, 
            include_labels=False,
            shuffle=False,
            n_epochs=1, args=args),
        checkpoint_path=ckpt_path)
    
    output_dir = os.path.join(args.model_dir, 'prediction_iter_'+args.model_iteration)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    previous_video_id = None
    for res in predict_generator:
        res['video_id'] = res['video_id'].decode("utf-8")
        if previous_video_id is None:
            print('Start inference for video: %s' % res['video_id'])
            previous_video_id = res['video_id']
        elif res['video_id'] != previous_video_id:
            print('Start inference for video: %s' % res['video_id'])
            previous_video_id = res['video_id']
            
        output_path = os.path.join(output_dir, 
            res['video_id']+'_'+str(res['predicted_time_points'][0]).zfill(5)+'.jpg')
        gazemap = np.reshape(res['ps'], args.gazemap_size)
        misc.imsave(output_path, gazemap)

  
    
  
  
  
  



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
