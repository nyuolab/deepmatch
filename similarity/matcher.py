import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json

import tables
from annoy import AnnoyIndex

SAVEDIR = 

class DeepMatcherSim(object):
    def __init__(self, simdata, filtered_inds, model, device='cpu', save_dir=None):
        """Initialize with necessary components.
        
        Model must contain loaded weights.
        
        """
        self.data = simdata
        self.model = model
        self.DEVICE = device

        # if not spec'd, save in the same directory as the input dataset's.
        if save_dir is None:
            self.save_dir = '/'.join(self.data_fn.split('/')[:-1])
        else:
            self.save_dir = save_dir

    def prepare_for_matching(self):
        """Prepare dataset for matching by converting to latent state and building index tree."""

        print("Embedding in latent space...")
        self._embed_in_latent_space()

        print("Building index trees...hold tight -_- eek.")
        self.build_index_trees()

        print("Finding nearest neighbors...")
        self.find_nearest_neighbors()

    def build_index_trees(self, force=False):
        """Construct index trees to identify preferences."""
        
        suffix = '_anntree.h5'
        # filename = self.db.filename.split('.h5')[0] + '_anntree.h5'
        filename = self.save_dir + self.filename + suffix
        self.index_tree = AnnoyIndex(self.model.build_params['latent']['dimensions'], 'euclidean')

        # Check to see if index tree has already been built.
        if os.path.exists(filename) and force == False:
            self.index_tree.load(filename)

        else: 
            ls_db = NISDatabase(self.latent_space_fn, 'case', state_inds=self.ls_state_inds)
            ls_db.change_state('control')
            for idx, sample in enumerate(ls_db):
                self.index_tree.add_item(idx, sample)
                if idx % 100000 == 0:
                    print(idx)

            print("Added items to index tree!")

            self.index_tree.build(100)
            self.index_tree.save(filename)

    def match(self):
        """Match the filtered patients."""
        matched = -128 * np.ones((self.ls_db.state_inds['case'].shape[0], 2), dtype='int')
        already_matched_controls = []
        already_matched_cases = []

        # For each level:
        for level in range(0, self.n_neighbors):
            init_matched = len(already_matched_cases)
            
            unique, idx, inv, cnts = np.unique(self.nearest_neighbors[:, 0, level], 
                                            return_index=True, 
                                            return_inverse=True, 
                                            return_counts=True)
            
            unique_matches_case = idx[cnts == 1]
            unique_matches_control = unique[cnts == 1] 

            # Identify unique matches and assign those                
            for unique_case, unique_control in zip(unique_matches_case, unique_matches_control):
                
                # Remove matches that have already been assigned (already_matched)
                if unique_control in already_matched_controls:
                    continue
                    
                if unique_case in already_matched_cases:
                    continue
                
                matched[unique_case, 0] = unique_control
                already_matched_controls.append(unique_control)
                already_matched_cases.append(unique_case)
            
             # Identify competing matches
            competing_controls = unique[cnts > 1]
                
            for competing_control in competing_controls:
                if competing_control in already_matched_controls:
                    continue
                
                competing_cases = np.where(self.nearest_neighbors[:, 0, level] == competing_control)
                competing_cases = np.setdiff1d(competing_cases, already_matched_cases)
                
                if len(competing_cases) > 0:
                    competing_dists = self.nearest_neighbors[competing_cases, 1, level]
                    winner = competing_cases[np.argmin(competing_dists)]
                    matched[winner, 0] = competing_control
                
                    already_matched_controls.append(competing_control)
                    already_matched_cases.append(winner)
                    
        #             print("Matched Case {0} with Control {1}".format(winner, competing_control))
                    
            print("Level {2} finished. Matched {0}. Remaining {1}.".format(len(already_matched_cases) - init_matched, self.nearest_neighbors.shape[0] - len(already_matched_cases), level))
            if len(already_matched_cases) == self.nearest_neighbors.shape[0]:
                break

        matches = {}
        matches['case'] = np.where(matched >= 0)[0].astype('int')
        matches['control'] = matched[matched >= 0].astype('int')

        self.matches = matches

    def find_nearest_neighbors(self):
        nearest_neighbors = []
        self.n_neighbors = 64

        self.ls_db.change_state('case')
        self.ls_db.set_batch_size(1)

        for idx, sample in enumerate(self.ls_db.iterator):
            nn_i = self.index_tree.get_nns_by_vector(sample.view(-1), self.n_neighbors, search_k=-1, include_distances=True)
            nearest_neighbors.append(nn_i)

        nearest_neighbors = np.array(nearest_neighbors)
        nearest_neighbors[:, 0, :] = nearest_neighbors[:, 0, :].astype('int')
            
        self.nearest_neighbors = nearest_neighbors

    def _embed_in_latent_space(self, force=False):
        """Embed the filtered cases and controls into the latent space."""

        suffix = '_latentspace.h5'
        inds_suffix = '_latentspacestateinds.json'

        self.latent_space_fn = self.save_dir + self.filename + suffix
        self.ls_state_inds_fn = self.save_dir + self.filename + inds_suffix

        ls_dim = self.model.build_params['latent']['dimensions']

        if os.path.exists(self.latent_space_fn) and force == False:
            ls_state_inds = json.load(open(self.ls_state_inds_fn, 'r'))
            self.ls_state_inds = {cohort : np.array(inds) for cohort, inds in ls_state_inds.items()}
            self.ls_db = NISDatabase(self.latent_space_fn, 'case', state_inds=self.ls_state_inds)

        else:
            with tables.open_file(self.latent_space_fn, 'w') as latent_space_db:
                latent_space_db.create_array('/', 'headers', self.db.headers)
                ls_arr = latent_space_db.create_earray('/', 'dataset', shape=(0, ls_dim), atom=tables.FloatAtom())
                ls_state_inds = {}

                start_idx = 0
                for cohort_type, inds in self.db.state_inds.items():
                    # Pick the cohort we want to iterate over and create the DataLoader to handle iteration.
                    self.db.change_state(cohort_type)
                    ls_state_inds[cohort_type] = np.arange(start_idx, inds.shape[0] + start_idx).astype('int')
                    
                    state_iter = DataLoader(self.db, batch_size=1000, pin_memory=True, num_workers=1)

                    for idx, cohort_batch in enumerate(state_iter):
                        cohort_batch = cohort_batch.to(self.DEVICE)
                        cohort_batch_ls = self.model.latent_representation(cohort_batch).detach().to('cpu')
                        ls_arr.append(np.array(cohort_batch_ls))
                    
                    start_idx += inds.shape[0]
            
            self.ls_db = NISDatabase(self.latent_space_fn, 'case', state_inds=ls_state_inds)
            self.ls_state_inds = ls_state_inds

            # Export ls_state_inds for use later.
            ls_state_inds = {cohort : list(inds) for cohort, inds in ls_state_inds.items()}
            json.dump(ls_state_inds, open(self.ls_state_inds_fn, 'w'), cls=NumpyEncoder)
            

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
