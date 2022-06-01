import tensorflow as tf
from tensorflow.keras import initializers, activations
import numpy as np

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234















class DenseNetHelper:

    attr_keys = ['weight_normal', 'scaling_info', 'scaling', 'initializer']

    def __init__(self, **helper_kw):
        # None for unavailable keys
        for key in DenseNetHelper.attr_keys:
            setattr(self, key, helper_kw.get(key))
                
    # ## activations ## #
    @property
    def activation(self):
        return {'tanh': activations.tanh,
                    'swish': activations.swish,
                        'relu':activations.relu, 
                            'softmax': self.softmax, 
                                'sigmoid': activations.sigmoid}
    
    @staticmethod
    def softmax(H):
        return activations.softmax(H, axis=-1)

    # ## weight and weight initializations ## #
    @property
    def init_variable(self):
        return {'xavier':initializers.GlorotNormal(), 
                    'random-normal':initializers.RandomNormal(mean=0.0, stddev = 1.0),
                        'uniform-variance':initializers.VarianceScaling(distribution = 'uniform'),
                            'normal-variance':initializers.VarianceScaling(distribution='untruncated_normal')}[self.initializer]

    # ## input scalings ## #
    def scale_min_max(self, inputs):
        return 2.0*(inputs - self.scaling_info[0])/(self.scaling_info[1] - self.scaling_info[0]) - 1.0
    
    def scale_mean_std(self, inputs):
        return (inputs - self.scaling_info[2])/self.scaling_info[3]
    
    # ## weight normalization methods #
    @property
    def weight_normals(self):
        return {'normal-yes': self.normalize_weights, 
                'normal-no': self.pass_on}
    
    @staticmethod
    def normalize_weights(W):
        return W/tf.norm(W, ord=2, axis=0, keepdims=True)
    
    @staticmethod
    def pass_on(W):
        return W

    def densenet_output(self, inputs, weights, biases, activation = 'relu'):
        H = {'minmax': self.scale_min_max, 
            'meanstd': self.scale_mean_std,
                'none': self.pass_on}[self.scaling](inputs)    
        for weight,bias in zip(weights, biases):
            weight = self.weight_normals[self.weight_normal](weight)
            H = self.activation[activation](tf.add(tf.matmul(H, weight), bias))
        return H
