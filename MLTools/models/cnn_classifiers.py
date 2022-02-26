import numpy as np
import pandas as pd 
from os import path, makedirs, environ 
from collections import namedtuple 
from datetime import datetime 
from functools import wraps 
from .. networks.architectures import CNNNet 
from .. datautils import generate 
import tensorflow.compat.v1 as tf
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.disable_v2_behavior()

class PlainImageClassifier:
    """
    A basic CNN-based image classifier
    """
    _optimizers = {'SGD':tf.train.MomentumOptimizer, 
                    'Adam': tf.train.AdamOptimizer}

    network_keys = ['num_filters', 
                    'dense_layers', 'cnn_strides', 'kernels', 'pool_type',
                        'pool_kernels', 'pool_strides', 'cnn_activation', 'dense_activation',
                             'initializer', 'bias_at_zero', 'input_shape']
    
    def __init__(self, data = None, **kwargs):
        self._data = data 
        self._loss = None 
        self.sess = tf.compat.v1.Session(config = tf.ConfigProto(allow_soft_placement = True, 
                    log_device_placement = True))
    
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        data_cond = hasattr(new_data, '_fields') and hasattr(new_data, '_asdict') and\
                    (new_data._fields in ('train_images', 'train_labels', 'test_images', 'test_labels', 'input_shape', 'targets'))
        if data_cond:
            self._data = new_data 
        else:
            raise TypeError('only accepting a namedtuple at this point')


    def configure_network(self, num_filters = [25, 50], dense_layers = [100], cnn_strides = (1,1), 
                kernels = (3,3), pool_type = 'max', pool_kernels = 2, pool_strides = 2, 
                    cnn_activation = 'relu', dense_activation = 'relu', initializer='xavier', bias_at_zero = False):
        net_kw = {key:val for key,val in locals().items() if key in self.network_keys}
        net_kw.update({'target_shape': self._data.target_shape})
        net_kw.update({'input_shape': self._data.input_shape})
        print('target shape is = ', self._data.target_shape)
        self.network = CNNNet(**net_kw)
        
    def configure_placeholder(self):
        w, h, c = self._data.input_shape
        self.train_input_pl = tf.placeholder(tf.float32, shape = [None, w, h, c])
        self.train_label_pl = tf.placeholder(tf.float32, shape = (None))
        self.test_input_pl = tf.placeholder(tf.float32, shape = [None, w, h, c])
        self.test_label_pl = tf.placeholder(tf.float32, shape = (None))
    
    @property
    def network_out_train(self):
        return self.network(self.train_input_pl)

    @property
    def network_out_test(self):
        return self.network(self.test_input_pl)
    
    def configure_loss(self, loss = 'sparse-crossentropy'):
        loss_func = {'sparse-crossentropy': tf.keras.losses.SparseCategoricalCrossentropy, 
                    'categorical-crossentropy': tf.keras.losses.CategoricalCrossentropy}[loss]()
        self.loss = tf.reduce_mean(loss_func(self.train_label_pl, self.network_out_train))
    

    def _setup_SGD(self, rate = 0.005, momentum = 0.9):
        return self._optimizers['SGD'](learning_rate = rate,
                         momentum = momentum)
    
    def _setup_adam(self, rate = 0.001):
        return self._optimizers['Adam'](learning_rate = rate)
                                                                                                                                                                                                                                                                           
    def configure_optimizer(self, optimizer = 'SGD', **kwargs):
        try:
            opt_object = {'SGD': self._setup_SGD, 
                'Adam': self._setup_adam}[optimizer](**kwargs)
            self.train_opt = opt_object.minimize(self.loss)
        except:
            raise NotImplementedError('the requested optimizer is not implemented yet')
         
    def configure(self):
        pass 

    def train(self, num_epochs = 1000, batch = None, output_every=50, save = None):
        """
        save is directory name for saving model outputs and the model itself
            for running from commandline it is recommended to specify this
        """

        if batch is None:
            batch = 1.0
        self.sess.run(tf.global_variables_initializer())
        self.train_loss = []
        self.train_accuracy = []
        self.elapsed_cycles = []
        for n_epoch in range(num_epochs):
            train_size = int(self._data.train_images.shape[0])
            train_batch_index = np.random.choice(train_size, int(train_size*batch), replace = False)
            # this step depends on the image type 
          #  train_inputs = np.expand_dims(self._data.train_images[train_batch_index], axis = 3) 
            train_inputs = self._data.train_images[train_batch_index]
            train_labels = self._data.train_labels[train_batch_index] 
            train_info = {self.train_input_pl: train_inputs, 
                            self.train_label_pl: train_labels}
            self.sess.run(self.train_opt, feed_dict = train_info)
            if n_epoch % output_every == 0 and n_epoch > 0:
                self.elapsed_cycles.append(n_epoch)
                train_loss, epoch_predictions = self.sess.run([self.loss, self.predictions], feed_dict = train_info)
                train_accuracy = self.accuracy(epoch_predictions, train_labels)
                self.train_accuracy.append(train_accuracy)
                self.train_loss.append(train_loss)                
                print('ran ', n_epoch, ' cycles and the loss value is ==> ', train_loss)
                if save is not None:
                    self._save_outputs(n_epoch, train_loss, train_accuracy, save)
    
    @property
    def predictions(self):
        return tf.nn.softmax(self.network_out_train)

    @staticmethod 
    def accuracy(logits, targets):
        predictions = np.argmax(logits, axis = 1)
        corrects = np.sum(np.equal(predictions, targets))
        return (corrects/predictions.shape[0])*100
    
    # #### useful utilities #### #
    def save_counter(_save_outputs):
        @wraps(_save_outputs)
        def save_wrapper(self, *args, **kwargs):
            save_wrapper.counter += 1
            if save_wrapper.counter == 1:
                save = args[3]
                date = datetime.today().strftime('%Y-%m-%d-%H-%m')
                out_dirname = path.join(save, type(self).__name__ + '_Train_Results_on_' + date)
                if not path.exists(out_dirname):
                    makedirs(out_dirname)
                save_wrapper.filename = path.join(out_dirname, 'Loss&Accuracy.csv')
                save_wrapper.append_mode = 'w'
                save_wrapper.header = True
            _save_outputs(self, *args, **kwargs)
        save_wrapper.counter = 0
        save_wrapper.filename = None
        save_wrapper.append_mode = 'a'
        save_wrapper.header =False
        return save_wrapper
                
    @save_counter 
    def _save_outputs(self, *args, **kwargs):
        cycles, loss, accuracy = args[0], args[1], args[2]
        out_frame = pd.DataFrame(np.array([cycles, loss, accuracy])[:, np.newaxis].T,
                     columns = ['cycles','loss','accuracy'])

        out_frame.to_csv(self._save_outputs.filename, sep =' ', mode = self._save_outputs.append_mode,
                 header = self._save_outputs.header, index = False,  float_format = '%.5f')

    def __call__(self):
        pass 

    @classmethod 
    def using_MNIST_images(cls, scale = True, one_hot = True, **kwargs):
        """
        depending on use case, call different MNIST generators
        use = 'classification' or 'image-to-image'
        """
        (train_images, train_code), (test_images, test_code) = generate.MNIST(use = 'classification', **kwargs)
        if scale:
            train_images = train_images/255
            test_images = test_images/255  
        # note that target size is needed to determine the size of last dense layer
        data = namedtuple('data', ['train_images', 'train_labels', 'test_images', 'test_labels', 'input_shape', 'targets'])
        # one-hot encode labels
        if one_hot:
            train_labels = np.zeros((train_code.shape[0], train_code.max() + 1), dtype = np.float32)
            train_labels[np.arange(0, train_code.shape[0]), train_code] = 1
            test_labels = np.zeros((test_code.shape[0], test_code.max() + 1), dtype = np.float32)
            test_labels[np.arange(0, test_code.shape[0]), test_code] = 1    
            data.train_labels = train_labels 
            data.test_labels = test_labels  
        else:
            data.train_labels = train_code
            data.test_labels = test_code 

        data.target_shape = len(set(train_code)) 
        data.train_images = np.expand_dims(train_images, axis = 3)
        data.test_images = np.expand_dims(test_images, axis = 3)
        data.input_shape = data.train_images.shape[1:]
        return cls(data, **kwargs)




        

