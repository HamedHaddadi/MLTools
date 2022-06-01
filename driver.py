
import argparse 
import yaml
import sys
from os import path
from MLTools.models.cnn_classifiers import PlainImageClassifier
from MLTools.models.autoencoders import DenseAutoEncoder, CNNAutoEncoder

    
def parse_file(inputfile):
    try:
        stream = open(inputfile, 'r')
        inputs = yaml.safe_load(stream)
    except FileExistsError:
        print('the input file does not exist')
    return inputs 

# #### Runs a PlainImageClassifier Model #### #
#: This class is a Plain CNN with Pool Layers that can be used for classification
# 
def run_plain_image_classifier(model_inputs):
    """
    generates inputs for the PlainImageClassifier
    then runs the model
    """
    
    dataset = model_inputs['data_configs']['source']
    del model_inputs['data_configs']['source']
    model = {'MNIST': PlainImageClassifier.using_MNIST_images,
                }[dataset](**model_inputs['data_configs'])

    model.configure_network(**model_inputs['network_configs'])
    model.configure_placeholder()
    model.configure_loss(**model_inputs['loss_configs'])
    model.configure_optimizer(**model_inputs['optimizer_configs'])
    model.train(**model_inputs['train_configs'])
# ########################################### #

# #### Runs a Dense AutoEncoder Model #### #
#: This class is a conventional AutoEncoder 
 
def run_dense_autoencoder(model_inputs):
    """
    generates inputs for a dense autoencoder model
    then runs the model
    """
    dataset = model_inputs['data_configs']['source']
    del model_inputs['data_configs']['source']
    model = {'MNIST': DenseAutoEncoder.using_MNIST_data
                    }[dataset](**model_inputs['data_configs'])
    model.configure_network(**model_inputs['network_configs'])
    model.configure_placeholder()
    model.configure_loss()
    model.configure_optimizer(**model_inputs['optimizer_configs'])
    model.train(**model_inputs['train_configs'])

# #### Runs a CNN AutoEncoder Model ##### #
def run_cnn_autoencoder(model_inputs):
    dataset = model_inputs['data_configs']['source']
    del model_inputs['data_configs']['source']
    print('data config information is ', model_inputs['data_configs'])
    model = {'MNIST': CNNAutoEncoder.using_MNIST_data
                    }[dataset](**model_inputs['data_configs'])
    model.configure_network(**model_inputs['network_configs'])
    model.configure_placeholder()
    model.configure_loss()
    model.configure_optimizer(**model_inputs['optimizer_configs'])
    model.train(**model_inputs['train_configs'])


if __name__ == '__main__':
    """
    reads input files and runs different models 
    useful for running models from commandlines using the yaml input files in MLTools.inputs
    """
    parser = argparse.ArgumentParser(description='input reader and module launch')
    parser.add_argument('-inputfile', nargs = 1, type = str, help='reads the input file name')
    inputs = parser.parse_args()
    model_inputs = parse_file(parser.parse_args().inputfile[0]) 
    model = model_inputs['model']
    {'PlainImage': run_plain_image_classifier, 
        'DenseAE': run_dense_autoencoder,
            'CNNAE': run_cnn_autoencoder}[model](model_inputs)   
    
    