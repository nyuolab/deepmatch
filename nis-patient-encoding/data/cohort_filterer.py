import numpy as np
import h5py
import torch
import torch.nn as nn

from functools import reduce

from data.data_loader import NISDatabase
from utils import feature_utils

ALLOWED_STATES = ['case', 'control']

class CohortFilterer(NISDatabase):
    def __init__(self, dataset_fn, filter_params):

        super().__init__(dataset_fn, 'full')
        
        self.filter_params = filter_params

    def filter_patients(self):
        """Build a case-control group based on very fundamental criteria."""

        if self.dataset is None:
            self.dataset = h5py.File(self.filename, 'r')['dataset']
        
        # Find feature indices belonging to specific criteria
        inclusion_info = self.filter_params['inclusion']
        # exclusion_info = self.filter_params['exclusion']
        case_control_info = self.filter_params['case_control']

        inclusion_inds = self.check_criteria(inclusion_info, case_control=False)
        # exclusion_inds = self.check_criteria(exclusion_info, case_control=False)
        case_inds, control_inds = self.check_criteria(case_control_info, case_control=True)

        filtered_inds = {}
        # inclusion_exclusion_inds = np.setdiff1d(inclusion_inds, exclusion_inds)
        filtered_inds['case'] = np.intersect1d(inclusion_inds, case_inds)
        filtered_inds['control'] = np.intersect1d(inclusion_inds, control_inds)

        return filtered_inds

    def find_feature(self, pattern):
        """Find the indices for a feature type in the dataset."""
        idxs = []
        for idx, header in enumerate(self.headers):
            header = header.decode('utf-8')
            lp = len(pattern)

            # Find continuations
            if header == pattern:
                idxs.append(idx)
            elif header[:lp] == pattern and header[lp] in [str(i) for i in range(0, 10)]:
                idxs.append(idx)

        return idxs
    
    def check_criteria(self, criteria, case_control=False):
        """Perform an intersection across all criteria that are upheld by the data."""

        if case_control:
            pts_meeting_criteria = {key : [] for key in ['case', 'control']}
        else:
            pts_meeting_criteria = []

        if len(criteria) == 0: # mostly for exclusion criteria.
            return np.array([])

        for name, criterion in criteria.items():
            print(name, criterion)
            feature_inds = self.find_feature(name)
            pts_meeting_criterion = self.search_by_chunk(self.dataset, feature_inds, criterion, case_control)
            
            if case_control:
                pts_meeting_criteria['case'].append(pts_meeting_criterion['case'])
                pts_meeting_criteria['control'].append(pts_meeting_criterion['control'])
            else:
                pts_meeting_criteria.append(pts_meeting_criterion)

        if case_control:
            return reduce(np.intersect1d, pts_meeting_criteria['case']), \
                    reduce(np.intersect1d, pts_meeting_criteria['control'])
        else:
            return reduce(np.intersect1d, pts_meeting_criteria)

    @staticmethod
    def search_by_chunk(dataset, feature_inds, criterion, case_control=False):
        count = 0
        chunk_size = 1000000

        if case_control:
            inds = {key : [] for key in ['case', 'control']}
        else:
            inds = []

        while count < dataset.shape[0]:
            print(count)
            # iteration
            if count + chunk_size > dataset.shape[0]:
                # count = dataset.shape[0]
                chunk_size = dataset.shape[0] - count
            
            # do something
            chunk = dataset[count:count+chunk_size, feature_inds].reshape(chunk_size, -1)
            
            match = np.empty((chunk.shape[0], len(criterion)))
            for i, criterion_i in enumerate(criterion):
                # print("Criterion: ", criterion_i)
                matches_i = (chunk == criterion_i).reshape(-1, chunk.shape[1]) # perform match
                match_i = (np.sum(matches_i, axis=1) > 0) # across all features
                match[:, i] = match_i
                        
            match = np.sum(match, axis=1) # assuming union, but @TODO make more generalizable

            if case_control:
                ind_case = np.where(match)[0] + count
                ind_control = np.where(match == 0)[0] + count
                inds['case'].extend(ind_case)
                inds['control'].extend(ind_control)

                # print(count, ind_case.shape[0], ind_control.shape[0])
                # print(count, len(inds['case']), len(inds['control']))

            else:
                inds_chunk = np.where(match)[0] + count # find indices
                inds.extend(inds_chunk)

            # finalize iteration
            count += chunk_size
            if count >= dataset.shape[0]:
                break

        return inds
