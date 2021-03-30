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
    
    'embedding' : {
        'dims' : 50
    },
    
    'encoding' : {
        'total_layers' : 1,
        'scale' : 4,
        'activation' : 'leaky_relu',
    },

    'latent' : {'dimensions' : 64},

    'decoding' : {
        'scale' : 4,
        'activation' : 'leaky_relu',
        'total_layers' : 1,
        'output_dims' : None
    }
}

BATCH_SIZE = 512
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2000

SAVE_PATH = '{0}/'.format(__file__.split('.')[0]) # remove.py and create a folder
print("Saving at {0}.".format(SAVE_PATH))

# DEVICE = torch.device('cuda')
DEVICE = torch.device('cuda:4')

"""
"""

# Main Script
def train():
    # Create the autoencoder.
    ae = AutoEncoder(DEFAULT_BUILD).to(DEVICE)

    # Instantiate the data loader.
    db = np.load(DATA_FOLDER + 'simulation_data_X_095.npy')
    db = db[:80000, :] # Training

    db_dl = DataLoader(db, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS, shuffle=True)

    # Instantiate the loss function.
    loss_function = nn.MSELoss().to(DEVICE)
    
    # Create our trainer.
    trainer = Trainer(ae, loss_function, LEARNING_RATE, db_dl, DEVICE)
    trainer.train(NUM_EPOCHS, SAVE_PATH)

if __name__ == '__main__':
    train()