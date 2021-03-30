import numpy as np
import h5py
import tables
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import feature_utils

ALLOWED_STATES = ['TRAIN', 'TEST', 'VAL']

class NISDatabase(torch.utils.data.Dataset):
    def __init__(self, filename, state, **kwargs):
        """ Manages loading of NIS database samples in a H5 file, particularly for training.
        """
        # super(NISDatabase, self).__init__()
        self.filename = filename
        self.dataset = None
        self.dataset_key = kwargs.get('dataset_key', 'dataset')
        
        # with h5py.File(self.filename, 'r', libver='latest', swmr=True) as file:
        with tables.open_file(self.filename, 'r') as file:
            # self.headers = file['headers'][:]
            self.headers = file.root.headers[:]

        self.state_inds = kwargs.get('state_inds', {})
        self.allowed_states = kwargs.get('allowed_states', [])
        self.pin_memory = kwargs.get('pin_memory', True)
        self.change_state(state)

        self.batch_size = kwargs.get('batch_size', 1000)
        self.iterator = DataLoader(self, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=0)

    def __getitem__(self, index):
        """ Load input data.

        Note that index really indexes into inds for training, test, val split.

        Also, loading data in worker processes and opening the HDF5 file *once*
        in __get_item__ ensures multithreading from the DataLoader class
        works appropriately. See
        https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        for full discussion.

        @TODO consider using torch.nn.data.Sampler instead of code below.

        """

        if self.dataset is None:
            # h5_file = h5py.File(self.filename, 'r', libver='latest', swmr=True)
            # self.dataset = h5_file[self.dataset_key]
            self.h5_file = tables.open_file(self.filename, 'r')
            self.dataset = getattr(self.h5_file.root, self.dataset_key)
            # h5_file.close()

        # print(self.h5_file)
        state_index = self.inds[index]
        return self.dataset[state_index]

    def __len__(self):
        """ Return length of dataset we're operating on.
        """
        return self.dataset_len

    def set_inds(self, inds):
        self.dataset_len = inds.shape[0]
        self.inds = inds

    def set_dataset_key(self, key):
        self.dataset_key = key
        self.dataset = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.iterator = DataLoader(self, batch_size=batch_size, pin_memory=self.pin_memory, num_workers=0)

    def change_state(self, state):
        """ Change attributes based on state of model.
        """

        if len(list(self.state_inds.keys())) == 0:
            # with h5py.File(self.filename, 'r') as file:
            with tables.open_file(self.filename, 'r') as file:
                # If we want the full dataset, this is how we specify it. 
                if state == 'full':
                    self.state_inds = {
                        # 'full' : np.arange(file[self.dataset_key].shape[0])
                        'full' : np.arange(getattr(file.root, self.dataset_key).shape[0])
                        }
                    self.allowed_states = ['full', ]

                else:
                    self.state_inds = {}
                    self.allowed_states = ALLOWED_STATES
                    for state_i in self.allowed_states:
                        # @TODO need to restructure our files so these inds are better nested...
                        # self.state_inds[state_i] = file[state_i][:] 
                        self.state_inds[state_i] = getattr(file.root, state_i)[:]

        else:
            # If we've already declared unique states, then just move on.
            if type(self.state_inds) == type({}):
                self.allowed_states = list(self.state_inds.keys())

                # State and dataset length info
        if state not in self.allowed_states:
            raise ValueError(f'{state} not found.')

        self.set_inds(self.state_inds[state])
