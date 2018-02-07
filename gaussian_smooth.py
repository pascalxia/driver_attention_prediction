# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:59:41 2017

@author: pasca
"""

#from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
import tensorflow.contrib.distributions as ds



class GaussianSmooth(Layer):

    def __init__(self, kernel_size, **kwargs):
        self.kernel_size = kernel_size
        #TODO: train this sigma
        self.sigma = 1.5
        super(GaussianSmooth, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        '''
        self.sigma = self.add_weight(name='sigma', 
                                      shape=(1,),
                                      initializer='ones',
                                      trainable=True)
        '''
        super(GaussianSmooth, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        kernel = self.calculate_kernel()
        return tf.nn.conv2d(x, kernel, strides=[1,1,1,1], padding='SAME')
        #return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def calculate_kernel(self):
        kernelSize = self.kernel_size
        xinds, yinds = np.unravel_index(range(kernelSize*kernelSize), 
                                        (kernelSize, kernelSize))
        inds = tf.constant((np.column_stack((xinds,yinds))-
                            [(kernelSize-1)/2, (kernelSize-1)/2]).astype(np.float32))
        
        loc = tf.zeros([2,])
        scale_diag = tf.multiply(self.sigma, tf.ones([2,]))
        
        mvn = ds.MultivariateNormalDiag(
            loc=loc,
            scale_diag=scale_diag)
        
        kernel = mvn.prob(inds)
        kernel = tf.reshape(kernel, (kernelSize, kernelSize, 1, 1))
        return kernel
        