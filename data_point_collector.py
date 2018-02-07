# -*- coding: utf-8 -*-
"""
@author: Ye Xia
"""

import os
import pickle

#data_dir = data/
#training set directory: data/training/
#validation set directory: data/validation/
#camera_images/
#gazemap_images/
#10_342.jpg


def read_datasets(data_dir, in_sequences=False):
    if in_sequences:
        pickle_filename = "data_point_names_in_sequences.pickle"
    else:
        pickle_filename = "data_point_names.pickle"
    
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        data_point_names = get_data_point_names(data_dir, in_sequences)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(data_point_names, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")
        with open(pickle_filepath, 'rb') as f:
            data_point_names = pickle.load(f)

    return data_point_names['training'], data_point_names['validation']


def get_data_point_names(data_dir, in_sequences=False):
    if not os.path.isdir(data_dir):
        print("Data directory '" + data_dir + "' not found.")
        return None
    
    data_point_names = {}
    directories = ['training', 'validation']
    for directory in directories:
        data_point_dict = {}
        image_path = os.path.join(data_dir, directory, 'camera_images')
        camera_points = set([f[:-4] for f in os.listdir(image_path) if f.endswith('.jpg')])
        
        image_path = os.path.join(data_dir, directory, 'gazemap_images')
        gazemap_points = set([f[:-4] for f in os.listdir(image_path) if f.endswith('.jpg')])
    
        data_points = camera_points.intersection(gazemap_points)
        
        if in_sequences:
            data_point_dict = {}
            for data_point in data_points:
                video_id = data_point.split('_')[0]
                group = data_point_dict.setdefault(video_id, [])
                group.append(data_point)
            data_point_names[directory] = \
                list(data_point_dict.values())
        else:
            data_point_names[directory] = list(data_points)
            
        no_of_videos = len(data_point_names[directory])
        print ('No. of %s videos: %d' % (directory, no_of_videos))
        
    return data_point_names  
    
    
    
    
    