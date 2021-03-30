import numpy as np
import h5py
import tables
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functools import reduce

from data.data_loader import NISDatabase
from data.cohort_filterer import CohortFilterer
from similarity.matchers import DeepMatcher
# from similarity.matchers import PropensityScoreMatcher
# from similarity.matchers import FuzzyMatcher
from utils import feature_utils
from utils.code_mappings import map_icd9_to_numeric

INCLUSION_INFO = {'DXCCS': [122, 109]}
CASE_CONTROL_INFO = {'DX': [map_icd9_to_numeric('42731'),]}
COHORT_FILTER_INFO = {'inclusion': INCLUSION_INFO, 'case_control': CASE_CONTROL_INFO}

DATA_FOLDER = '/home/aisinai/work/repos/nis_patient_encoding/data/raw/'

DATASETS = {
    'filtering' : f'{DATA_FOLDER}NIS_Pruned.h5',
    'matching' : f'{DATA_FOLDER}NIS_2012_2014_proto_emb_v2.h5',
    'eval_input': f'{DATA_FOLDER}NIS_2012_2014_proto_emb_v2.h5',
    'eval_output': f'{DATA_FOLDER}NIS_2012_2014_proto_emb_v2.h5',
}

MATCHERS = {
    # 'propensity' : ,
    'deep' : DeepMatcher,
    # 'fuzzy' : FuzzyMatcher,
}

class CohortBuilder():
    """Manages initial cohort construction.

    This class takes in the full DataFrame with raw ICD-9 codes to find the indices that map to particular cohorts. 
    These indices may then be used by downstream classes in more pruned/filtered datasets for use in the AE system.

    """

    def __init__(
        self,   
        datasets=DATASETS, 
        filter_params=COHORT_FILTER_INFO):

        """Manages loading of NIS database samples in a H5 file. 
        
        Note: reimplemented here, but @TODO to refactor this entire class structure.
        
        """
        # super().__init__(filename, 'TRAIN')

        self.filter_params = filter_params
        self.datasets = datasets
        
        matcher_results = {matcher : {'case': [], 'control': []} for matcher in MATCHERS.keys()}
        self.inds = {
            'filtered': {'case': [], 'control': []}, # from full dataset to those included in current analysis
            'matched': {'case': [], 'control': []} # from included in current analysis to matched
            }

        self.filtered = False
        self.matched = False

    def prepare_for_match(self):
        raise NotImplementedError

    def filter_patients(self, sample=None):
        """Build a case-control group based on very fundamental criteria."""

        filter_dataset = self.datasets['filtering']

        filterer = CohortFilterer(filter_dataset, self.filter_params)
        self.inds['filtered'] = filterer.filter_patients()

        if sample:
            case_alias = self.inds['filtered']['case']
            control_alias = self.inds['filtered']['control']
            self.inds['filtered']['case'] = np.random.choice(case_alias, int(sample * case_alias.shape[0]), replace=False)
            self.inds['filtered']['control'] = np.random.choice(control_alias, int(sample * control_alias.shape[0]), replace=False)

        self.filtered = True

    def match_patients(self, model, device='cpu', save_dir=None):
        """Match patients for each matcher."""

        if self.filtered == False:
            raise ValueError("Must perform filtering before matching, silly!")

        matcher_input = {
            'dataset' : self.datasets['matching'],
            'state_inds' : self.inds['filtered']
        }

        self.matchers = {}

        for matcher_name, matcher_class in MATCHERS.items():
            matcher = matcher_class(matcher_input['dataset'], matcher_input['state_inds'], model, device=device, save_dir=save_dir)

            self.matchers[matcher_name] = {}
            self.matchers[matcher_name]['matcher'] = matcher
            
            matcher.prepare_for_matching()
            matcher.match()

            self.matchers[matcher_name]['matches'] = matcher.matches

    def set_match_keys(self, matcher_name):
        matches = self.matchers[matcher_name]['matches']
        self.inds['matched']['case'] = self.inds['filtered']['case'][matches['case']]
        self.inds['matched']['control'] = self.inds['filtered']['control'][matches['control']]
    
    def _create_batch_for_eval(self, matcher_name, feature_info):
        batches = {}

        state_inds = {}
        self.set_match_keys(matcher_name)

        db = NISDatabase(DATASETS['eval_input'], 'case', state_inds=self.inds['matched'])

        for cohort_type, cohort_inds in self.inds['matched'].items():
            cohort_presenting = {}

            db.change_state(cohort_type)
            cohort_dl = DataLoader(db, batch_size=1000, pin_memory=True, num_workers=1)

            for idx, batch in enumerate(cohort_dl):
                target = self._isolate_features(batch, feature_info)
                
                for feature, recon in target.items():
                    if feature not in list(cohort_presenting.keys()):
                        recon_np = np.array(recon.detach().to('cpu'))
                        cohort_presenting[feature] = recon_np
                    else:
                        recon_np = np.array(recon.detach().to('cpu'))
                        cohort_presenting[feature] = np.vstack((cohort_presenting[feature], recon_np))
            
            batches[cohort_type] = cohort_presenting

        return batches

    @staticmethod
    def _isolate_features(x, feature_info):
        """Modify the features as needed."""

        # Initalize stores
        ground_truth = {}
        # x = torch.Tensor(x)

        for embedding, embedding_layer in feature_info['embedding'].items():
            features_to_use = embedding_layer['feature_idx']
            batch_emb = x[:, features_to_use].to(torch.long)
            
            ground_truth[embedding] = nn.functional.one_hot(batch_emb, num_classes=embedding_layer['num_classes'])
            ground_truth[embedding] = (torch.sum(ground_truth[embedding], axis=1) > 0).to(torch.float)

        for one_hot_name, one_hot in feature_info['one_hots'].items():
            nc = one_hot['num_classes']
            one_hot_encoded = nn.functional.one_hot(x[:, one_hot['feature_idx']].to(torch.long), num_classes=nc)
            
            ground_truth[one_hot_name] = one_hot_encoded

        for cont_name, continuous_feature in feature_info['continuous'].items():
            cont = x[:, continuous_feature['feature_idx']].view(-1, 1)
            ground_truth[cont_name] = cont

        return ground_truth