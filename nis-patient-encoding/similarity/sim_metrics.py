import numpy as np
from scipy.stats import chisquare
import pandas as pd

import h5py
import tables

from functools import reduce

from data.data_loader import NISDatabase
from data.cohort_filterer import CohortFilterer
from similarity.matchers import DeepMatcher
from utils import feature_utils
from utils.code_mappings import map_icd9_to_numeric

def perform_ttest(case_data, control_data):
    """Perform T-Test on the case/control data."""

    t_test_vals = pd.DataFrame(index=case_data.keys())
    t_tests = {}

    for (_, case), (feature_name, control) in zip(case_data.items(), control_data.items()): 
        t_tests[feature_name] = {}
        tfn = t_tests[feature_name] # shorthand
        
        # Find number of class representatives
        ft_0 = np.sum(control, axis=0)
        ft_1 = np.sum(case, axis=0)
        
        # Remove nonzero classes
        ft_nonzero = np.array([[ft_0i, ft_1i] for ft_0i, ft_1i in zip(ft_0, ft_1) if ft_1i != 0]).astype('int')
        # t_tests[feature_name]['case_control'] = pd.DataFrame(np.vstack((ft_0, ft_1)).T, columns=['CONTROL', 'CASE']).rename_axis(feature_name)
        
        # Compute the chi-sq value and p-value
        tfn['chisq'], tfn['pval'] = chisquare(ft_nonzero[:, 0], ft_nonzero[:, 1])

        tfn['p_value'] = tfn['pval']
        tfn['chisq'] = tfn['chisq']

        t_test_vals.loc[feature_name, 'P-Value'] =  tfn['pval']
        t_test_vals.loc[feature_name, 'Chi-Statistic'] =  tfn['chisq']
        t_test_vals.loc[feature_name, 'Type'] = 'Categorical'

    return t_test_vals

def calculate_JSD(case_data, control_data):
    """Calculate in the latent space."""
    
    if type(case_data) != type(np.array()):
        raise ValueError('Expecting a numpy array!')

    

    