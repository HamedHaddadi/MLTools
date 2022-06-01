
from os import path 
import numpy as np
import tensorflow as tf
import gzip 


# ############################# #
# . generates various datasets  #
# ############################# #

reshape_flat = lambda array, length, shape: np.reshape(array, (length, shape*shape))
reshape_image = lambda array, length, shape: np.reshape(array, (length, shape, shape, 1))

def MNIST_from_file(data_path = '/Users/hamedhaddadi/Documents/ML/MLTools/datasets',
         train_filename = 'train-images-idx3-ubyte.gz',
             test_filename = 't10k-images-idx3-ubyte.gz', split = 0.8, format = 'flat'): 
    """
    split: train/validation split
    outputs: train, validation, test datasets
    size of train dataset: 48000 x 784 (28 x 28)
    size of validation dataset: 12000 x 784
    size of test dataset: 10000 x 784
    """
    image_size = 28
    num_images = 60000
    num_tests = 10000

    print('the format is = ', format)

    train_file = path.join(data_path, train_filename)
    f = gzip.open(train_file, 'r')
    f.read(16)
    buffer = f.read(image_size*image_size*num_images)
    all_images = np.frombuffer(buffer, dtype= np.uint8).astype(np.float32)

    all_images = {'flat': reshape_flat, 
                    'image':reshape_image}[format](all_images, num_images, image_size)
    
    # train/validation splits
    train_index = np.random.choice(num_images, round(split*num_images), replace = False)
    train_images = all_images[train_index]
    validation_index = [i for i in range(num_images) if i not in train_index]
    validation_images = all_images[validation_index]


    # test data
    test_file = path.join(data_path, test_filename)
    f = gzip.open(test_file, 'r')
    f.read(16)
    buffer = f.read(image_size*image_size*num_tests)
    test_images = np.frombuffer(buffer, dtype = np.uint8)

    test_images = {'flat': reshape_flat, 
                    'image': reshape_image}[format](test_images, num_tests, image_size)

    # normalize
    train_images = train_images*(1/(train_images.max() - train_images.min()))
    test_images = test_images*(1/(test_images.max() - test_images.min()))
    validation_images = validation_images*(1/(validation_images.max() - validation_images.min()))

    return train_images, validation_images, test_images 

def MNIST_from_keras(**kwargs):
    return tf.keras.datasets.mnist.load_data()

def MNIST(use = 'classification', **kwargs):
    return {'classification': MNIST_from_keras, 
                    'image-to-image':MNIST_from_file}[use](**kwargs)

