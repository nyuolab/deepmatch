import os
os.chdir('/home/aisinai/work/repos/deepmatch-simulation')

import tables
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.autoencoder import AutoEncoder
from trainer.trainer import Trainer

# Create Global Variables
DATA_FOLDER = 'data/'

DEFAULT_BUILD = {
    'data' : {
        'input_features' : 500
    },
    
    'encoding' : {
        'total_layers' : 2,
        'scale' : 3,
        'activation' : 'leaky_relu',
    },

    'latent' : {'dimensions' : 32},

    'decoding' : {
        'scale' : 3,
        'activation' : 'leaky_relu',
        'total_layers' : 2,
        'output_dims' : None
    }
}

BATCH_SIZE = 512
NUM_WORKERS = 4
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100000

SAVE_PATH = '{0}/'.format(__file__.split('.')[0]) # remove.py and create a folder
print("Saving at {0}.".format(SAVE_PATH))

# DEVICE = torch.device('cuda')
DEVICE = torch.device('cuda:0')

"""
"""

# Main Script
def train():
    # Create the autoencoder.
    ae = AutoEncoder(DEFAULT_BUILD).to(DEVICE)

    # Instantiate the data loader.
    db = np.load(DATA_FOLDER + 'simulation_data_X.npy')
    db = db[:80000, :] # Training

    db_dl = DataLoader(db, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS, shuffle=True)

    # Instantiate the loss function.
    loss_function = nn.MSELoss().to(DEVICE)
    
    # Create our trainer.
    trainer = Trainer(ae, loss_function, LEARNING_RATE, db_dl, DEVICE)
    trainer.train(NUM_EPOCHS, SAVE_PATH)

if __name__ == '__main__':
    train()