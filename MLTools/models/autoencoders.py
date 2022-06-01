
from abc import ABCMeta, abstractmethod 
import numpy as np
import pandas as pd 
from os import path, makedirs, getcwd 
from datetime import datetime 
import matplotlib.pyplot as plt
from collections import namedtuple
from functools import wraps 
from .. networks.architectures import DenseAENet, CNNAENet
from .. datautils import generate 
import tensorflow.compat.v1 as tf 

tf.disable_v2_behavior()

class AutoEncoder(metaclass = ABCMeta):

    _optimizers = {'Adam': tf.train.AdamOptimizer, 
                        'GD': tf.train.GradientDescentOptimizer}
    _opt_keys = ['optimizer', 'rate']
    _train_keys = ['num_epochs', 'batch', 'save']

    """
    The base class for all autoencoders
    implementation using static graphs in Tensorflow 
    If Tensorflow 2, the eager computation must be disabled
    """
    def  __init__(self, data = None, **kwargs):
        self._data  = data
        self.loss = None 
        self.sess = tf.compat.v1.Session(config = tf.ConfigProto(allow_soft_placement = True,
                     log_device_placement=True))
    
    @property
    def data(self):
        return self._data 
    
    @data.setter 
    def data(self, data):
        """
        data must be a named tuple
            contaning train, validation, test and input_features
        """
        if hasattr(data, '_fields') and hasattr(data, '_asdict'):
            self._data = data
        else:
            raise TypeError("the input for data must be a namedtuple") 
    
    @abstractmethod 
    def configure_network(self):
        ...

    def configure_loss(self):
        """
        Note: AE compares with itself
        """
        self.loss = tf.reduce_mean(tf.square(self.train_pl - self.predict))
    
    def _setup_adam(self, rate = 0.001, **kwargs):
        optimizer = self._optimizers['Adam'](learning_rate = rate)
        self.train_opt = optimizer.minimize(self.loss)
    
    def _setup_GD(self, rate = 0.001, **kwargs):
        optimizer = self._optimizers['GD'](learning_rate = rate)
        self.train_opt = optimizer.minimize(self.loss)

    def configure_optimizer(self, optimizer = 'Adam', **kwargs):
        {'Adam': self._setup_adam, 
            'GD': self._setup_GD}[optimizer](**kwargs)
    
    @property
    def predict(self):
        return self.network(self.train_pl)

    def method_counter(method):
        @wraps(method)
        def method_wrapper(self, *args, **kwargs):
            method_wrapper.counter += 1
            method(self, *args, **kwargs)
        method_wrapper.counter = 0
        return method_wrapper 

    # ##### setup place holders ##### #  
    configure_placeholder = lambda self: setattr(self, 'train_pl', tf.placeholder(tf.float32, shape = [None, self._data.input_features]))

    # ##### Main Training Method ##### #
    def train(self, num_epochs = 500, batch = None,
                 output_every=100, save = None, num_predictions = 4):
        """
        num_figure_predictions = 0 means that no figure is generated during training
            in calls to _save_outputs
        """
        if batch is None:
            batch = 1.0
        self.train_loss = []
        self.validation_loss = []
        self.num_epochs = num_epochs  

        self.sess.run(tf.global_variables_initializer())
        save_kw = {}
        for n_epoch in range(num_epochs):
            train_size = self._data.train.shape[0]
            train_batch_index = np.random.choice(train_size, int(batch*train_size), replace = False)
            train_inputs = self._data.train[train_batch_index]
            self.sess.run(self.train_opt, feed_dict = {self.train_pl:train_inputs})
            train_loss = self.sess.run(self.loss, feed_dict = {self.train_pl: train_inputs})
            print('training loss is ', train_loss)
            self.train_loss.append(train_loss)
            save_kw['n_epoch'] = n_epoch
            save_kw['train_loss'] = train_loss
            
            if 'validation' in self._data._fields:    
                validation_size = self._data.validation.shape[0]
                validation_batch_index = np.random.choice(validation_size, int(batch*validation_size), replace = False)
                validation_inputs = self._data.validation[validation_batch_index]
                validation_loss = self.sess.run(self.loss, feed_dict={self.train_pl:validation_inputs})
                self.validation_loss.append(validation_loss)
                save_kw['validation_loss'] = validation_loss 

            if n_epoch % output_every == 0 and n_epoch != 0:
                print('ran ', n_epoch, ' epochs out of ', num_epochs)
                self._save_outputs(save=save, num_predictions = num_predictions, **save_kw)
        
        print('Training finished ...')

    # #### methods to save and output results #### #
    def save_counter(_save_outputs):
        @wraps(_save_outputs)
        def save_wrapper(self, *args, **kwargs):
            save_wrapper.counter += 1
            if save_wrapper.counter == 1:
                date =  datetime.today().strftime('%Y-%m-%d-%H-%m')
                _root = {True:getcwd(),
                             False:kwargs['save']}[kwargs['save'] is None]
                save_dirname = path.join(_root, type(self).__name__ + '_Train_Results_On_' + date)
                if not path.exists(save_dirname):
                    makedirs(save_dirname)
                save_wrapper.save_dirname = save_dirname 
                save_wrapper.loss_filename = path.join(save_dirname, 'Loss.csv')
                save_wrapper.append_mode = 'w'
                save_wrapper.header = True 
            _save_outputs(self, *args, **kwargs)
        save_wrapper.counter = 0
        save_wrapper.loss_filename = None 
        save_wrapper.append_mode = 'a'
        save_wrapper.header = False 
        return save_wrapper 
        
    @save_counter 
    def _save_outputs(self, n_epoch = None, train_loss = None,
             validation_loss = None, save=None, num_predictions = 0):
        
        if save is None:
            print('outputs will be saved in the current directory')

        if n_epoch is None and train_loss is None and validation_loss is None:
            print('all loss values are None')
        else:
            results = np.array([n_epoch, train_loss, validation_loss])[:, np.newaxis].T
            results_df = pd.DataFrame(results, columns = ['cycles','train_loss','validation_loss'])
            results_df.to_csv(self._save_outputs.loss_filename, 
                                    sep = ' ', header = self._save_outputs.header,
                                        mode = self._save_outputs.append_mode, index = False, 
                                            float_format = '%.5f')
            if num_predictions != 0:
                self.test_model(num_predictions = num_predictions, save = True)

    @staticmethod 
    def generate_images(inputs, outputs, save):
        num_images = len(inputs)
        size = int(inputs[0].shape[0]**0.5)
        fig, axs = plt.subplots(nrows = num_images, ncols = 2, figsize = (8, 8*num_images))
        axs = axs.ravel()
        for img_count in range(num_images):
            axs[img_count*2].pcolor(inputs[img_count].reshape(size, size))
            axs[img_count*2 + 1].pcolor(outputs[img_count].reshape(size, size))
            fig.savefig(save)
        
    # ######## Predictions and Testing the Model ######### #
    #. prediction methods
    def test_model(self, num_predictions = 4, save = True):
        test_inputs = self._data.test[np.random.choice(len(self._data.test), num_predictions)]
        test_outputs = self.sess.run(self.predict, feed_dict = {self.train_pl:test_inputs})
        if save:
            save_figname = path.join(self._save_outputs.save_dirname,
                 'Images_' + str(self._save_outputs.counter) + '.png')
        self.generate_images(test_inputs, test_outputs, save = save_figname)

        
    # ##########  Useful training track methods ########### #
    def plot_loss(self):
        fig, axs = plt.subplots(figsize = (6,6))
        epochs = np.arange(0, len(self.train_loss))
        axs.plot(epochs, np.array(self.train_loss), '-', linewidth = 2, color = 'red', label = 'training')
        if len(self.validation_loss) != 0:
            axs.plot(epochs, np.array(self.validation_loss), '-', linewidth = 2, color = 'blue', label = 'validation')
        axs.set_xlabel('number of epochs')
        axs.set_ylabel('loss')
    
    close_session = lambda self: self.sess.close()

    @abstractmethod 
    def __call__(self):
        ...
    
    # #### factories to initialize class using different datasets #### #
    @classmethod 
    def using_MNIST_data(cls, use = 'image-to-image', format = 'flat', **kwargs):
        """
        format can be 'flat' or 'image'
            'flat' returns images of (?,28x28) and 
            'image' returns images of (?,28,28,1)
        """
        train, validation, test = generate.MNIST(use = use, format = format, **kwargs)
        input_features = train.shape[1]
        input_shape = train.shape[1:]
        data = namedtuple('data', ['train', 'validation', 'test', 'input_features', 'input_shape'])
        data.train = train 
        data.validation = validation
        data.test = test
        data.input_features = input_features 
        data.input_shape = input_shape 
        return cls(data, **kwargs)

# ##################################################### #
# . AutoEncoder using dense networks with a contraction #
# ##################################################### #
class DenseAutoEncoder(AutoEncoder):
    """
    AutEncoder using dense network
    """
    network_keys = ['compression', 'weight_normal', 
                    'scaling_info', 'scaling', 'activations', 'initializer']
    
    # ##### setup place holders ##### #  
    configure_placeholder = lambda self: setattr(self, 'train_pl', tf.placeholder(tf.float32, shape = [None] + list(self._data.input_shape)))    
    
    def configure_network(self, compression = 8, scaling = 'none', activations = 'relu-sigmoid', 
                        initializer = 'uniform-variance', weight_normal = False, scaling_info = 'none'):
        """
        compression is the ratio of input_features to the latent layer:
            example: 784 inputs features; and compression = 8; latent layer: 98 neurons
        increasing the compression makes a longer network 
        """
        net_kw = {key:val for key,val in locals().items() if key in self.network_keys}
        net_kw.update({'input_features':self._data.input_features})
        self.network = DenseAENet(**net_kw)
    
    @AutoEncoder.method_counter     
    def configure(self, **kwargs):
        config_kw = {key:val for key,val in kwargs.items() if key in self.network_keys}
        self.configure_network(**config_kw)
        self.configure_placeholder()
        self.configure_loss()
        opt_kw = {key:val for key,val in kwargs.items() if key in self._opt_keys}
        self.configure_optimizer(**opt_kw)

    # #################### #
    # .    __call__        #
    # #################### # 
    def __call__(self, **kwargs):
        if self.configure.counter == 0:
            config_kw = {key:val for key,val in kwargs.items() if key in self.dense_keys + self._opt_keys}
            self.configure(**config_kw)        
        train_kw = {key:val for key,val in kwargs.items() if key in self._train_keys}
        self.train(**train_kw)

# ##################################################### #
# . Denoising AutoEncoder                               #
# ##################################################### 
class CNNAutoEncoder(DenseAutoEncoder):
    """
    AutoEncoder using Conv2D and Conv2D_Transpose blocks
    """
    network_keys = ['filters', 'kernels',
                     'activation', 'initializer',
                     'bias_at_zero', 'strides']
    
    def configure_network(self, filters = [], kernels = [2,2], activation = 'relu', 
            initializer = 'normal-variance', bias_at_zero = True, strides = [2,2]):
        network_kw = {key:val for key,val in locals().items() if key in self.network_keys}
        network_kw.update({'input_shape':self._data.input_shape})
        self.network = CNNAENet(**network_kw)

    # ############################### #
    # methods for testing the outputs #
    # ############################### #
    @staticmethod
    def generate_images(inputs, outputs, save):
         num_images = len(inputs)
         size = inputs[0].shape[0]
         fig, axs = plt.subplots(nrows = num_images, ncols = 2, figsize = (8, 8*num_images))
         axs = axs.ravel()
         for img_count in range(num_images):
             axs[img_count*2].pcolor(inputs[img_count].reshape(size, size))
             axs[img_count*2 + 1].pcolor(outputs[img_count].reshape(size, size))
             fig.savefig(save)          

        
        
        
        

        

        
    
    
    



     
