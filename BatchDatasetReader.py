# -*- coding: utf-8 -*-
"""
@author: Ye Xia
"""

import numpy as np
import scipy.misc as misc
import random
from keras.preprocessing.image import ImageDataGenerator
import feather
import ut
import os.path as path


MAX_SEED = 99999

class BatchDataset:
    data_dir = ''
    data_point_names = []
    files = []
    image_names = []
    annotation_names = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    annotation_threshold = None
    

    def __init__(self, data_dir, data_point_names, image_size=None,
                 feature_name='vgg',
                 annotation_threshold=None,
                 weight_data=False):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image [height, width] - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        self.data_dir = data_dir
        self.data_point_names = data_point_names
        random.shuffle(self.data_point_names)
        self.annotation_threshold = annotation_threshold
        self.image_size = image_size
        self.image_generator = ImageDataGenerator(width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            rotation_range=10)
        self.annot_generator = ImageDataGenerator(width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            rotation_range=10,
                                            fill_mode='constant')
        if weight_data:
             df = feather.read_dataframe(path.join(data_dir,'sampling_weights.feather'))
             df.set_index('fileName', inplace=True)
             self.data_weight_df = df
             self.prepare_sampling_weights()

        else:
            self.data_weight_df = None 
            self.data_weights = None
            
        self.seed = 0
        if feature_name=='vgg':
            self.feature_folder = 'image_features'
        elif feature_name=='bdd':
            self.feature_folder = 'image_features_bdd'
        elif feature_name=='alexnet':
            self.feature_folder = 'image_features_alexnet'
                
    
    
    def prepare_sampling_weights(self):
        if not isinstance(self.data_point_names[0], list):
            self.data_weights = self.data_weight_df.loc[self.data_point_names, 'weight'].values
        else:
            self.data_weights = np.zeros(len(self.data_point_names))
            for i, seq in enumerate(self.data_point_names):
                weights = self.data_weight_df.loc[seq, 'weight'].values
                self.data_weights[i] = np.sum(weights)
        #normalize
        self.data_weights = self.data_weights/np.sum(self.data_weights)
    
    
    def read_image(self, data_point_name):
        return misc.imread(path.join(self.data_dir,'camera_images',
                                     data_point_name+'.jpg'))
        
    def read_annotation(self, data_point_name):
        annotation = misc.imread(path.join(self.data_dir,'gazemap_images',
                                           data_point_name+'.jpg'))
        return annotation[:,:,0]
    
    def read_feature_map(self, data_point_name):
        return np.load(path.join(self.data_dir,self.feature_folder,data_point_name+'.npy'))
    
    def get_images(self, data_point_names, augment=None):
        desired_size = self.image_size
        images = []
        for name in data_point_names:
            images.append(self.read_image(name))
        images = np.array(images)
        if desired_size is not None and images.shape[1:3] != desired_size:
            images = np.array([misc.imresize(image, desired_size, 
                                             interp='bilinear')
                                for image in images])
        if augment == 'random':
            images = self.image_generator.flow(images, batch_size=len(images), 
                                         seed=self.seed).__next__()
        elif augment == 'same':
            for i, img in enumerate(images):
                one_img = self.image_generator.flow(np.array([img]), batch_size=1, 
                                              seed=self.seed).__next__()
                images[i] = one_img[0]
        if augment is not None:
            images = images.astype(np.uint8)
        return images
    
    
    def get_annotations(self, data_point_names, desired_size=None, augment=None):
        if desired_size is None:
            desired_size = self.image_size
        annotations = []
        for name in data_point_names:
            annotations.append(self.read_annotation(name))
        annotations = np.array(annotations)
        if desired_size is not None and annotations.shape[1:3] != desired_size:
            annotations = np.array([ut.resize_distribution(annotation, desired_size) 
                for annotation in annotations])
        
        #normalize
        annotations = annotations.astype(np.float)
        for i in range(len(annotations)):
            annot_sum = np.sum(annotations[i])
            if annot_sum != 0:
                annotations[i] /= annot_sum
            else:
                annotations[i][:] = 1 / annotations[i].size
        
        if self.annotation_threshold is not None:
            annotations = self.binarize(annotations)
        else:
            annotations = np.expand_dims(annotations, axis=3)
            
        if augment == 'random':
            annotations = self.annot_generator.flow(annotations, 
                                              batch_size=len(annotations), 
                                              seed=self.seed).__next__()
        elif augment == 'same':
            for i, annot in enumerate(annotations):
                one_annot = self.annot_generator.flow(np.array([annot]), batch_size=1, 
                                                seed=self.seed).__next__()
                annotations[i] = one_annot[0]
        if augment is not None:
            #Normalize again
            annotations = annotations.astype(np.float)
            for i in range(len(annotations)):
                annot_sum = np.sum(annotations[i])
                if annot_sum != 0:
                    annotations[i] /= annot_sum
                else:
                    annotations[i][:] = 1 / annotations[i].size
                    
        return annotations
    
    def get_feature_maps(self, data_point_names):
        feature_maps = []
        for name in data_point_names:
            feature_maps.append(self.read_feature_map(name))
        feature_maps = np.array(feature_maps)
        return feature_maps

    def binarize(self, raw_annotations):
        height, width = raw_annotations.shape[1:3]
        annotations = np.zeros(raw_annotations.shape[0:3], dtype=np.float32)
        annotations[raw_annotations*height*width>self.annotation_threshold] = 1.0
        annotations = np.expand_dims(annotations, axis=3)
        return annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset
    
    def next_batch(self, batch_size, augment=None):
        if augment is not None:
            self.seed = random.randint(0, MAX_SEED)
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.data_point_names):
            #get the rest of the dataset
            batch1 = self.data_point_names[start:]
            #get some of the beginning of the dataset to complete the batch
            batch2 = self.data_point_names[0:(self.batch_offset-len(self.data_point_names))]
            #concatenate them
            batch = batch1 + batch2
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            random.shuffle(self.data_point_names)
            if self.data_weight_df is not None:
                self.prepare_sampling_weights()
            # reset batch offset
            self.batch_offset = 0
        else:
            end = self.batch_offset
            batch = self.data_point_names[start:end]
        return batch
         
    
    def next_batch_in_seqs(self, batch_size, n_steps=None, augment=None):
        batch = self.next_batch(batch_size, augment)
        if n_steps is not None:
            batch = self.truncate(batch, n_steps)
        return batch
        
    
    #return a random batch of images and annotations
    def random_batch(self, batch_size, augment=None):
        if augment is not None:
            self.seed = random.randint(0, MAX_SEED)
        
        batch = np.random.choice(self.data_point_names, size=batch_size,
                                 p=self.data_weights)
        
        return batch
    
    def random_batch_in_seqs(self, batch_size, n_steps=None, augment=None):
        batch = self.random_batch(batch_size, augment)
        if n_steps is not None:
            batch = self.truncate(batch, n_steps)
        return batch
    
    def get_feature_maps_in_seqs(self, data_point_names_in_seqs):
        feature_maps = []
        for seq in data_point_names_in_seqs:
            seq_maps = self.get_feature_maps(seq)
            feature_maps.append(seq_maps)
        feature_maps = self.sequence_padding(feature_maps)
        return feature_maps
            
    def sequence_padding(self, sequence_list):
        lengths = []
        for seq in sequence_list:
            lengths.append(seq.shape[0])
        num_steps = max(lengths)
        for i in range(len(sequence_list)):
            seq = sequence_list[i]
            unit_shape = seq.shape[1:]
            padded = np.concatenate((np.zeros((num_steps-lengths[i],) + 
                                              unit_shape),
                                     sequence_list[i]),
                           axis=0)
            sequence_list[i] = padded
        sequence_array = np.array(sequence_list)
        return sequence_array
    
    def truncate(self, data_point_names_in_seqs, n_steps):
        for i, seq in enumerate(data_point_names_in_seqs):
            if len(seq) > n_steps:
                if self.data_weight_df is None:
                    ind = random.randint(0, len(seq) - n_steps)
                else:
                    weights = self.data_weight_df.loc[seq, 'weight'].values
                    cum_weights = np.cumsum(weights)
                    n = len(weights)
                    #calculate the weights of the starting points
                    weights = np.insert(cum_weights[n_steps:]-cum_weights[:-n_steps], 0, cum_weights[n_steps-1])
                    weights = weights/np.sum(weights)
                    ind = np.random.choice(range(n-n_steps+1), size=1, p=weights)[0]                    
                data_point_names_in_seqs[i] = seq[ind:(ind+n_steps)]
        return data_point_names_in_seqs
    
    def get_annotations_in_seqs(self, data_point_names_in_seqs, 
                                desired_size=None, augment=None):
        annotations = []
        for seq in data_point_names_in_seqs:
            seq_annotations = self.get_annotations(seq, desired_size,
                                                   augment)
            annotations.append(seq_annotations)
        annotations = self.sequence_padding(annotations)
        return annotations
    
    def get_images_in_seqs(self, data_point_names_in_seqs, augment=None):
        images = []
        for seq in data_point_names_in_seqs:
            seq_images = self.get_images(seq, augment)
            images.append(seq_images)
        images = self.sequence_padding(images)
        images = images.astype(np.uint8)
        return images
    
    def get_data_weights_in_seqs(self, data_point_names_in_seqs):
        data_weights = []
        for seq in data_point_names_in_seqs:
            seq_data_weights = self.get_data_weights(seq)
            data_weights.append(seq_data_weights)
        data_weights = self.sequence_padding(data_weights)
        data_weights = np.array(data_weights)
        return data_weights
        
    
             

    
