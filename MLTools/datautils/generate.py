
from os import path 
import numpy as np
import tensorflow as tf
import gzip 


# ############################# #
# . generates various datasets  #
# ############################# #

def MNIST_from_file(data_path = '/Users/hamedhaddadi/Documents/PINNPDE/MLTools/datasets',
         train_filename = 'train-images-idx3-ubyte.gz',
             test_filename = 't10k-images-idx3-ubyte.gz', split = 0.8, format = 'data'): 
    """
    split: train/validation split
    outputs: train, validation, test datasets
    size of train dataset: 48000 x 784 (28 x 28)
    size of validation dataset: 12000 x 784
    size of test dataset: 10000 x 784
    """
    image_size = 28
    num_train = 60000
    train_file = path.join(data_path, train_filename)
    f = gzip.open(train_file, 'r')
    f.read(16)
    buffer = f.read(image_size*image_size*num_train)
    all_images = np.frombuffer(buffer, dtype= np.uint8).astype(np.float32).reshape(num_train, image_size*image_size)

    # train/validation splits
    train_index = np.random.choice(len(all_images), round(split*len(all_images)), replace = False)
    train_images = all_images[train_index]

    validation_index = [i for i in range(len(all_images)) if i not in train_index]
    validation_images = all_images[validation_index]

    # test data
    num_test = 10000
    test_file = path.join(data_path, test_filename)
    f = gzip.open(test_file, 'r')
    f.read(16)
    buffer = f.read(image_size*image_size*num_test)
    test_images = np.frombuffer(buffer, dtype = np.uint8).reshape(num_test, image_size*image_size)

    # normalize
    train_images = train_images*(1/(train_images.max() - train_images.min()))
    test_images = test_images*(1/(test_images.max() - test_images.min()))
    validation_images = validation_images*(1/(validation_images.max() - validation_images.min()))

    if 'image' in format:
        train_images  = np.reshape(train_images, (num_train, image_size, image_size, 1))
        validation_images = np.reshape(validation_images, (len(validation_index), image_size, image_size, 1))
        test_images = np.reshape(test_images, (num_test, image_size, image_size, 1))

    return train_images, validation_images, test_images 

def MNIST_from_keras(**kwargs):
    return tf.keras.datasets.mnist.load_data()

def MNIST(use = 'classification', **kwargs):
    return {'classification': MNIST_from_keras, 
                    'image-to-image':MNIST_from_file}[use](**kwargs)

