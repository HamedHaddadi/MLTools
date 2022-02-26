
from abc import ABCMeta, abstractmethod 
import tensorflow as tf 
from tensorflow.keras import initializers, activations 
from math import log 
import numpy as np 
from functools import wraps 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#from . nnhelpers import DenseNetHelper

# ############################################ #
# . Abstract base class for all Deep Networks  #
# .  works for both CNNs and Dense             #  
# ############################################ #

class NeuralNet(metaclass = ABCMeta):
    NN_keys = ['weight_normal', 'scaling_info', 'scaling', 'initializer']
    def __init__(self, **network_kw):
        for key in NeuralNet.NN_keys:
            setattr(self, key, network_kw.get(key))

    # ## activations ## #
    @property
    def activation_functions(self):
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
                     'truncated-normal': tf.random.truncated_normal,
                        'uniform-variance':initializers.VarianceScaling(distribution = 'uniform'),
                            'normal-variance':initializers.VarianceScaling(distribution='untruncated_normal'), 
                                   'zero': tf.zeros}[self.initializer]

   # ## input scalings ## #
    def scale_min_max(self, inputs):
        return 2.0*(inputs - self.scaling_info[0])/(self.scaling_info[1] - self.scaling_info[0]) - 1.0
    
    def scale_mean_std(self, inputs):
        return (inputs - self.scaling_info[2])/self.scaling_info[3]
    
    # ## weight normalization methods 33 #
    @property
    def weight_normals(self):
        return {True: self.normalize_weights, 
                False: self.pass_on}
    
    @staticmethod
    def normalize_weights(W):
        return W/tf.norm(W, ord=2, axis=0, keepdims=True)
    
    @staticmethod
    def pass_on(W, *args, **kwargs):
        return W
    
    @abstractmethod
    def _define_parameters(self):
        ...
    
    @abstractmethod
    def network(self):
        ...
    
    @abstractmethod 
    def __call__(self):
        ... 

# ################################# #
# .           Dense Network Class   #
# ################################# #

class DenseNet(NeuralNet):
    def __init__(self, **network_kw):
        super(DenseNet, self).__init__(**network_kw)
       # helper_kw = {key:val for key,val in network_kw.items() if key in DenseNet.NN_keys}
       # self.nn_helper = DenseNetHelper(**helper_kw)
        self.network_kw = {key:val for key,val in network_kw.items() if key not in DenseNet.NN_keys}
        self.layers = None 
        activations = network_kw['activations'].split('-')
        if len(activations) == 1:
            self.activations = activations[0]
        else:
            self.activations = activations
        self.weights = []
        self.biases = []  
        self.network()
    
    def _define_parameters(self):
        pass 

    def network(self):
        self._define_parameters()
        for num in range(len(self.layers) - 1):
            self.weights.append(self.init_variable(shape = (self.layers[num], self.layers[num + 1])))
            self.biases.append(tf.Variable(tf.zeros((1, self.layers[num + 1]), dtype=tf.float32))) 

    def densenet_output(self, inputs, weights, biases, activation = 'relu'):
        H = {'minmax': self.scale_min_max, 
            'meanstd': self.scale_mean_std,
                'none': self.pass_on}[self.scaling](inputs)    
        for weight,bias in zip(weights, biases):
            weight = self.weight_normals[self.weight_normal](weight)
            H = self.activation_functions[activation](tf.add(tf.matmul(H, weight), bias))
        return H
    
    def __call__(self):
        pass 

    
# ############################## #
#  Autoencoder with a bottleneck #
# ############################## #

class DenseAENet(DenseNet):
    """
    For auto-encoder using a dense network:
        => number of features in input 
            determines the number of neurons in the first layer
        => compression factor determines the number of layers
    """

    def __init__(self, **network_kw):
        super(DenseAENet, self).__init__(**network_kw)

    def _define_parameters(self):
        self.x_features = self.network_kw['input_features']
        num_en_layers = int(log(self.network_kw['compression'], 2))
        encode_neurons = [int(self.x_features/num) for num in range(1, num_en_layers + 1)]
        latent_neurons = [int(self.x_features/self.network_kw['compression'])]
        decode_neurons = encode_neurons[::-1]
        encoder_layers = encode_neurons + latent_neurons 
        decoder_layers = decode_neurons
        self.len_encoder = len(encoder_layers)
        self.len_decoder = len(decoder_layers)
        self.layers = encoder_layers + decoder_layers
        
    def network(self):
        self._define_parameters()
        for num in range(len(self.layers) - 1):
            self.weights.append(tf.Variable(self.init_variable(shape = (self.layers[num], self.layers[num + 1])), dtype = tf.float32))
            self.biases.append(tf.Variable(self.init_variable(shape = (1, self.layers[num + 1])), dtype=tf.float32))

    def __call__(self, inputs):
        encoder_out = self.densenet_output(inputs, self.weights[:self.len_encoder], self.biases[:self.len_encoder], activation = self.activations[0])
        decoder_out = self.densenet_output(encoder_out, self.weights[self.len_encoder:], self.biases[self.len_encoder:], activation = self.activations[1])
        return decoder_out   

# ######################## #
# Convolutional Networks   #
# ######################## #

class CNNNet(NeuralNet):
    """
    class for CNN based networks
    Note:
        num_filters: a list of integers; len(num_filters) = number of CNN blocks 
        dense_layers = a list of weights: exaple [50, 100] creates 
            a dense layer of 50 and 100 neurons respectively at the end of CNN layers
        ===> NOTE: for a classification example the outout of last layers is the number of categories
            so, its weight size is = [weight of layer N - 1, # of targets]
        ===> NOTE: if dense_layers is not None, this transition layer will be added automatically.  
        cnn_strides: a list of [s_w, s_h] format: currently uniform strides for all layers
        kernels: a list of [w,h] format
        cnn_activation: activation function for CNN layers; 
        ===> NOTE: self.activation_functions is the bank of all available activations defined for this class
                  they are defined in th base class
        pool_type = 'max', 'avg' or 'none' 
        pool_kernels: an integer; it will be multiplied by the number of CNN blocks
        pool_strides: an integer; will be multiplied by the number of CNN blocks
             currently only uniform strides for all pool layers 
        ===> NOTE: for more architectures of CNN-Pool more methods must be added to the class
        ===> NOTE: input shape is auto-detected by calling network 
    """
    attr_keys = ['num_filters', 'dense_layers', 'cnn_strides', 'kernels', 'pool_type',
                        'pool_kernels', 'pool_strides', 'cnn_activation', 'dense_activation',
                             'initializer', 'bias_at_zero', 'target_shape', 'input_shape']

    def __init__(self, **network_kw):
        for key in network_kw.keys():
            if key in CNNNet.attr_keys:
                setattr(self,key, network_kw.get(key))
        # add a final layer for transition to target labels
        if self.dense_layers is not None:
            self.dense_layers += [0]
        else:
            self.dense_layers = []
        self.cnn_weights = []
        self.cnn_biases = []
        self.dense_weights = []
        self.dense_biases = []
        self.pool_kernels = [self.pool_kernels]*len(self.num_filters)
        self.pool_strides = [self.pool_strides]*len(self.num_filters)
        self._define_parameters()

    
    def network(self):
        pass 
    
    # place a decorator here for checking dimnsions correctly
    def _define_parameters(self):
        """
        define the weights and biases
        as opposed to a dense net, weights nd biases are defined in 
        this method; which are then used by define_network in conv2d layers
        """
        w, h = self.kernels 
        for count in range(len(self.num_filters)):
            if count == 0:
                weight_shape = (w, h, self.input_shape[2], self.num_filters[count])
            else:
                weight_shape = (w, h, self.num_filters[count - 1], self.num_filters[count])
            
            self.cnn_weights.append(tf.Variable(self.init_variable(shape = weight_shape),
                                 dtype = tf.float32))

            if self.bias_at_zero:
                self.cnn_biases.append(tf.Variable(tf.zeros([self.num_filters[count]]), dtype=tf.float32))
            else:
                self.cnn_biases.append(tf.Variable(self.init_variable(shape = [self.num_filters[count]])))
        
     #   if self.input_shape is not None:
            # determine the input shape to the first dense layer
        width_compression = 1
        height_compression = 1
        for num in range(len(self.num_filters)):
            width_compression *= self.cnn_strides[0]
            height_compression *= self.cnn_strides[1]
        if 'none' not in self.pool_type:
            for stride in self.pool_strides:
                width_compression *= stride 
                height_compression *= stride
            
        dense_input_shape = (self.input_shape[0]//width_compression)*\
                (self.input_shape[1]//height_compression)*self.num_filters[-1]
        for count, num_weights in enumerate(self.dense_layers):
            if count == len(self.dense_layers) - 1:
                weight_shape = (dense_input_shape, self.target_shape) 
                bias_shape = (1, self.target_shape)
            else:
                weight_shape = (dense_input_shape, num_weights)
                bias_shape = (1, num_weights)
            self.dense_weights.append(tf.Variable(self.init_variable(shape = weight_shape, dtype = tf.float32)))
            self.dense_biases.append(tf.Variable(self.init_variable(shape = bias_shape), dtype = tf.float32))                    
            dense_input_shape = num_weights 
            
    def pool(self, *args, **kwargs):
        return {'max':tf.nn.max_pool, 
                'avg':tf.nn.avg_pool, 
                    'none': self.pass_on}[self.pool_type](*args, **kwargs)

    def __call__(self, inputs):
        for count, params in enumerate(zip(self.cnn_weights, self.cnn_biases)):
            weight, bias = params[0], params[1]
            net_out = tf.nn.conv2d(inputs, weight, strides = [1, self.cnn_strides[0], self.cnn_strides[1], 1], padding = 'SAME')
            net_out = tf.nn.bias_add(net_out, bias)
            net_out = self.activation_functions[self.cnn_activation](net_out)
            net_out = self.pool(net_out, self.pool_kernels[count], self.pool_strides[count], 'SAME', data_format = 'NHWC')
            inputs = net_out 
        # add dense layers
        if len(self.dense_layers) > 0:
            # flatten CNN results 
            cnn_shape = tf.shape(net_out)
            flat_shape = cnn_shape[1]*cnn_shape[2]*cnn_shape[3]
            # input to dense is a flattened array
            dense_in = tf.reshape(net_out, (cnn_shape[0], flat_shape))
            for num_dense in range(len(self.dense_weights) - 1):
                net_out = self.activation_functions[self.dense_activation](tf.add(tf.matmul(dense_in, self.dense_weights[num_dense]),
                     self.dense_biases[num_dense]))
                dense_in = net_out
            net_out = tf.add(tf.matmul(net_out, self.dense_weights[-1]), self.dense_biases[-1])
        
        return net_out
            
        
            











            





















        




    
          



