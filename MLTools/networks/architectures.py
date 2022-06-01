
from abc import ABCMeta, abstractmethod 
import tensorflow as tf 
from tensorflow.keras import initializers, activations 
from math import log 
import numpy as np 
from functools import wraps 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#from . nnhelpers import DenseNetHelper
activation_functions = {'tanh': activations.tanh,
                    'swish': activations.swish,
                        'relu':activations.relu, 
                            'softmax': activations.softmax, 
                                'sigmoid': activations.sigmoid}

init_variable = {'xavier':initializers.GlorotNormal(), 
                  'random-normal':initializers.RandomNormal(mean=0.0, stddev = 1.0),
                  'truncated-normal': tf.random.truncated_normal,
                   'uniform-variance':initializers.VarianceScaling(distribution = 'uniform'),
                   'normal-variance':initializers.VarianceScaling(distribution='untruncated_normal'), 
                    'zero': tf.zeros}

def pass_on(W, *args, **kwargs):
    return W

def pool(pool_type, *args, **kwargs):
    return {'max': tf.nn.max_pool, 
                'avg': tf.nn.avg_pool, 
                    'none':pass_on}[pool_type](*args, **kwargs)

# ############################################ #
# . Abstract base class for all Deep Networks  #
# .  works for both CNNs and Dense             #  
# ############################################ #
class NeuralNet(metaclass = ABCMeta):
    NN_keys = ['weight_normal', 'scaling_info', 'scaling', 'initializer']
    def __init__(self, **network_kw):
        for key in NeuralNet.NN_keys:
            setattr(self, key, network_kw.get(key))

   # ## input scalings ## #
    def scale_min_max(self, inputs):
        return 2.0*(inputs - self.scaling_info[0])/(self.scaling_info[1] - self.scaling_info[0]) - 1.0
    
    def scale_mean_std(self, inputs):
        return (inputs - self.scaling_info[2])/self.scaling_info[3]
    
    # ## weight normalization methods 33 #
    @property
    def weight_normals(self):
        return {True: self.normalize_weights, 
                False: pass_on}
    
    @staticmethod
    def normalize_weights(W):
        return W/tf.norm(W, ord=2, axis=0, keepdims=True)
    
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
            self.weights.append(init_variable[self.initializer](shape = (self.layers[num], self.layers[num + 1])))
            self.biases.append(tf.Variable(tf.zeros((1, self.layers[num + 1]), dtype=tf.float32))) 

    def densenet_output(self, inputs, weights, biases, activation = 'relu'):
        H = {'minmax': self.scale_min_max, 
            'meanstd': self.scale_mean_std,
                'none': pass_on}[self.scaling](inputs)    
        for weight,bias in zip(weights, biases):
            weight = self.weight_normals[self.weight_normal](weight)
            H = activation_functions[activation](tf.add(tf.matmul(H, weight), bias))
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
            self.weights.append(tf.Variable(init_variable[self.initializer](shape = (self.layers[num], self.layers[num + 1])), dtype = tf.float32))
            self.biases.append(tf.Variable(init_variable[self.initializer](shape = (1, self.layers[num + 1])), dtype=tf.float32))

    def __call__(self, inputs):
        encoder_out = self.densenet_output(inputs, self.weights[:self.len_encoder], self.biases[:self.len_encoder], activation = activations[0])
        decoder_out = self.densenet_output(encoder_out, self.weights[self.len_encoder:], self.biases[self.len_encoder:], activation = activations[1])
        return decoder_out   

# ######################## #
# Convolutional Networks   #
# ######################## #
class CNNNet:
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
            self.cnn_weights.append(tf.Variable(init_variable[self.initializer](shape = weight_shape), dtype = tf.float32))
            if self.bias_at_zero:
                self.cnn_biases.append(tf.Variable(tf.zeros([self.num_filters[count]]), dtype=tf.float32))
            else:
                self.cnn_biases.append(tf.Variable(init_variable[self.initializer](shape = [self.num_filters[count]])))
        
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
            self.dense_weights.append(tf.Variable(init_variable[self.initializer](shape = weight_shape, dtype = tf.float32)))
            self.dense_biases.append(tf.Variable(init_variable[self.initializer](shape = bias_shape), dtype = tf.float32))                    
            dense_input_shape = num_weights 
            
    def __call__(self, inputs):

        for count, params in enumerate(zip(self.cnn_weights, self.cnn_biases)):
            weight, bias = params[0], params[1]
            net_out = tf.nn.conv2d(inputs, weight, strides = [1, self.cnn_strides[0], self.cnn_strides[1], 1], padding = 'SAME')
            net_out = tf.nn.bias_add(net_out, bias)
            net_out = activation_functions[self.cnn_activation](net_out)
            net_out = pool(self.pool_type, net_out, self.pool_kernels[count], self.pool_strides[count], 'SAME', data_format = 'NHWC')
            inputs = net_out 

        # add dense layers
        if len(self.dense_layers) > 0:
            # flatten CNN results 
            cnn_shape = tf.shape(net_out)
            flat_shape = cnn_shape[1]*cnn_shape[2]*cnn_shape[3]
            # input to dense is a flattened array
            dense_in = tf.reshape(net_out, (cnn_shape[0], flat_shape))
            for num_dense in range(len(self.dense_weights) - 1):
                net_out = activation_functions[self.dense_activation](tf.add(tf.matmul(dense_in, self.dense_weights[num_dense]),
                     self.dense_biases[num_dense]))
                dense_in = net_out
            net_out = tf.add(tf.matmul(net_out, self.dense_weights[-1]), self.dense_biases[-1])
        
        return net_out

# ######################## #
#          CNN AE          #
# ######################## #
class CNNAENet:
    attr_keys = ['filters', 'kernels', 
                    'strides', 'activation',
                     'initializer', 'bias_at_zero', 
                        'input_shape']

    def __init__(self, **network_kw):
        for key in network_kw.keys():
            if key in CNNAENet.attr_keys:
                setattr(self, key, network_kw.get(key))    
        
        self.encoder_weights = []
        self.encoder_biases = []

        self.decoder_weights = []
        self.decoder_biases = []

        self._define_parameters()

    def _define_parameters(self):
        """
        defines the weights for the encoder pass
         decoder is built by reversing these lists
        """
        w,h = self.kernels 
        
        for count in range(len(self.filters)):
            if count == 0:
                weight_shape = (w, h, self.input_shape[2], self.filters[count])
            else:
                weight_shape = (w, h, self.filters[count - 1], self.filters[count])
            self.encoder_weights.append(tf.Variable(init_variable[self.initializer](shape = weight_shape), dtype = tf.float32))
            
            if self.bias_at_zero:
                self.encoder_biases.append(tf.Variable(tf.zeros([self.filters[count]]), dtype = tf.float32))
            else:
                self.encoder_biases.append(tf.Variable(init_variable[self.initializer]
                        (shape = [self.filters[count]]), dtype = tf.float32))

        for weight in self.encoder_weights[::-1]:
            self.decoder_weights.append(weight)
            if self.bias_at_zero:
                self.decoder_biases.append(tf.Variable(tf.zeros([weight.get_shape().as_list()[2]])))
            else:
                self.decoder_biases.append(tf.Variable(init_variable[self.initializer]
                            (shape = weight.get_shape().as_list()[2]), dtype = tf.float32))
    
    def encoder(self, inputs):
        shapes = []
        layer_in = inputs
        for weight, bias in zip(self.encoder_weights, self.encoder_biases):
            shapes.append(layer_in.get_shape().as_list())
            filter_out = tf.nn.conv2d(layer_in, weight, strides = [1, self.strides[0], self.strides[1], 1], padding = 'SAME')
            layer_out = activation_functions[self.activation](tf.nn.bias_add(filter_out, bias))
            layer_in = layer_out
        return layer_out, shapes 

    def decoder(self, encoder_out, shapes):
        trans_in = encoder_out 
        for weight, bias, shape in zip(self.decoder_weights, self.decoder_biases, shapes[::-1]):
            trans_shape = [tf.shape(encoder_out)[0], shape[1], shape[2], shape[3]]
            trans_out = tf.nn.conv2d_transpose(trans_in, weight, trans_shape, 
                            strides = [1, self.strides[0], self.strides[1], 1], padding = 'SAME')
            trans_out = activation_functions[self.activation](tf.nn.bias_add(trans_out, bias))
            trans_in = trans_out 
        return trans_out 
    

    def __call__(self, inputs):
        encoder_out, shapes = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out, shapes)
        return decoder_out 

