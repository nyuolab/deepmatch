import numpy as np
import pandas as pd
from numba import jit

import torch
import torch.nn as nn

def get_vocab_size(series):
    """ Return the vocab size of a categorical series.

    Arguments:
        series: np.ndarray or similar to find the vocab size of.

    Returns:
        int: length of vocabulary
        vocab: elements in the vocabulary

    """

    if series.ndim > 1:
        series = series.reshape(-1, )

    vocab = np.unique(series, return_counts=False)

    return len(vocab), vocab

def degranularize_icd9(series, depth_to_remove=2):
    """ Degranularize an ICD9 code.

    This digit returns the nth-parent for ICD-9 codes in series.

    """
    scale_factor = 10 ** (depth_to_remove)
    return np.floor(series / scale_factor) * scale_factor

def calc_output_dims(DEFAULT_BUILD):
    output_dims = np.sum([embedding['num_classes'] * len(embedding['feature_idx']) for embedding in DEFAULT_BUILD['features']['embedding'].values()])
    output_dims += np.sum([catfeat['num_classes'] for catfeat in DEFAULT_BUILD['features']['one_hots'].values()])
    output_dims += np.sum([1 for contfeat in DEFAULT_BUILD['features']['continuous'].values()])
    DEFAULT_BUILD['decoding']['output_dims'] = output_dims

def one_hot_encode(features, num_classes, return_tensor=False):
    if type(features) == type(np.array([])):
        features_tensor = torch.Tensor(features).to(torch.int64)

    one_hot = nn.functional.one_hot(features_tensor, num_classes=num_classes)

    if return_tensor:
        return one_hot
    else:
        return np.array(one_hot.detach())

#### FEATURE MATCHING

def find_feature(headers, pattern):
    """Find the indices for a feature type in the dataset."""
    idxs = []
    for idx, header in enumerate(headers):
        header = header.decode('utf-8')
        lp = len(pattern)

        # Find continuations
        if header == pattern:
            idxs.append(idx)
        elif header[:lp] == pattern and header[lp] in [str(i) for i in range(0, 10)]:
            idxs.append(idx)

    return idxs

def find_matches(feature_chunk, criterion):
    """Find whether features in a given column match a particular criterion."""
    match = np.empty((feature_chunk.shape[0], len(criterion)))

    for i, criterion_i in enumerate(criterion):
        matches_i = (feature_chunk == int(criterion_i)).reshape(-1, feature_chunk.shape[1]) # perform match
        match_i = (np.sum(matches_i, axis=1) > 0) # across all features
        match[:, i] = match_i
                
    match = np.sum(match, axis=1) # assuming union, but @TODO make more generalizable
    return match

def get_chadsvasc(db):

    CHADSVASC_SCORES = {
        'CHF' : {'DXCCS' : [108, ]},
        'Hypertension' : {'DXCCS' : [98, ]},
        'Diabetes' : {'DXCCS' : [49, 50]},
        'Stroke' : {'DXCCS' : [109, 112]},
        'Vascular Disease' : {'DX' : [41200], 'DXCCS' : [114, 115, 116]}, # in appropriate ICD9 format as spec'd in HCUP data elements
    }

    scores = np.zeros((db.dataset_len))

    for idx, (feature_chunk, index) in enumerate(zip(db.iterator, db.inds)):
        # print('Retrieving CHADS-VASC, chunk: ', idx)
        # Convert chunk to numpy array.
        chunk = np.array(feature_chunk.detach())

        # Create empty array to store feature_chunks.
        scores_chunk = np.zeros((chunk.shape[0]))

        # Age
        age_idx = find_feature(db.headers, 'AGE')
        scores_chunk += (chunk[:, age_idx] >= 65).reshape(-1).astype('int')
        scores_chunk += (chunk[:, age_idx] >= 75).reshape(-1).astype('int')

        # Gender
        gender_idx = find_feature(db.headers, 'FEMALE')
        scores_chunk += (chunk[:, gender_idx]).reshape(-1).astype('int')

        # Comorbidities
        for comorbidity, criterion in CHADSVASC_SCORES.items():
            for field, criterion_i in criterion.items():
                criterion_i_idx = find_feature(db.headers, field)
                matches = find_matches(chunk[:, criterion_i_idx], criterion_i)
                scores_chunk += matches.reshape(-1).astype('int')

        # Store final scores in massive scores array.
        scores[idx * db.batch_size:idx * db.batch_size + feature_chunk.shape[0]] = scores_chunk
        
        return scores

def get_comorbidities(db, comorbidities, exclusion=False):

    # trait = np.zeros((db.dataset_len, len(comorbidities.keys())))
    comorbs = pd.DataFrame(index=np.arange(db.dataset_len), columns=comorbidities.keys())

    for idx, (feature_chunk, index) in enumerate(zip(db.iterator, db.inds)):
        # print('Retrieving comorbidities, chunk: ', idx)
        # Convert chunk to numpy array.
        chunk = np.array(feature_chunk.detach())

        # Comorbidities
        for comorb_name, trait_criteria in comorbidities.items():
            for i, (field, criterion_i) in enumerate(trait_criteria.items()):
                criterion_i_idx = find_feature(db.headers, field)
                comorb_flags = find_feature(db.headers, 'CHRON') 

                if field == 'DX':
                    # Since we have mutual exclusivitiy already created, need to include 'CHRON' flags in our search.
                    criterion_i_idx.extend(comorb_flags)
                    
                else:
                    # Probably DXCCS codes which are not acct'd for chronicity
                    chunk[:, criterion_i_idx] = chunk[:, criterion_i_idx] * (chunk[:, comorb_flags] > 0).astype('uint8')

                matches = find_matches(chunk[:, criterion_i_idx], criterion_i)

                if i == 0:
                    trait_chunk = (matches > 0).reshape(-1).astype('int')
                else:
                    trait_chunk += (matches > 0).reshape(-1).astype('int')

            # Store final scores in massive scores array.
            s_idx = idx * db.batch_size
            e_idx = s_idx + chunk.shape[0] - 1

            comorbs.loc[s_idx:e_idx, comorb_name] = (trait_chunk > 0).astype('int')
        
        
    return comorbs

def get_severity(db, severities_details):
    
    severities = pd.DataFrame(index=np.arange(db.dataset_len), columns=severities_details.keys())
    for idx, (feature_chunk, index) in enumerate(zip(db.iterator, db.inds)):
        # print('Retrieving severities, chunk: ', idx)
        # Convert chunk to numpy array.
        chunk = np.array(feature_chunk.detach())

        # Comorbidities
        for severity_name, severity_level_criteria in severities_details.items():
            for severity_level, trait_criteria in severity_level_criteria.items():

                severity_level = int(severity_level)
                
                for j, (field, criterion_i) in enumerate(trait_criteria.items()):
                    criterion_i_idx = find_feature(db.headers, field)
                    comorb_flags = find_feature(db.headers, 'CHRON')

                    if field == 'DX':
                        # Since we have mutual exclusivitiy already created, need to include 'CHRON' flags in our search.
                        criterion_i_idx.extend(comorb_flags)
                        
                    else:
                        # Probably DXCCS codes which are not acct'd for chronicity
                        chunk[:, criterion_i_idx] = chunk[:, criterion_i_idx] * (chunk[:, comorb_flags] > 0).astype('uint8')

                    matches = find_matches(chunk[:, criterion_i_idx], criterion_i)
                    
                    if j == 0:
                        trait_chunk = (matches > 0).reshape(-1).astype('int')
                    else:
                        trait_chunk += (matches > 0).reshape(-1).astype('int')
                    
                tcg0 = (trait_chunk > 0)

                if severity_level == 1:
                    severity_level_chunk = (tcg0).astype('int') # Increment severity_level for the chunk.
                else:
                    severity_level_chunk[tcg0] = severity_level * (tcg0[tcg0]).astype('int') # Increment severity_level for the chunk.
                

            # Store final scores in massive scores array.
            s_idx = idx * db.batch_size
            e_idx = s_idx + chunk.shape[0] - 1

            severities.loc[s_idx:e_idx, severity_name] = severity_level_chunk
                
    return severities
    
def find_trait(db, trait_criteria):

    trait = np.zeros((db.dataset_len))

    for idx, (feature_chunk, index) in enumerate(zip(db.iterator, db.inds)):
        # Convert chunk to numpy array.
        chunk = np.array(feature_chunk.detach())

        # Comorbidities
        for field, criterion_i in trait_criteria.items():
            criterion_i_idx = find_feature(db.headers, field)
            matches = find_matches(chunk[:, criterion_i_idx], criterion_i)
            trait_chunk = (matches > 0).reshape(-1).astype('int')

        # Store final scores in massive scores array.
        trait[idx * db.batch_size:idx * db.batch_size + chunk.shape[0]] = trait_chunk
        
    return trait

def retrieve_elements_from_db(db, col, num_cols='single'):
    
    if num_cols == 'multiple':
        out = np.empty((db.inds.shape[0], len(col)))
        stack_func = np.vstack
    else:
        out = np.empty((db.inds.shape[0], ))
        stack_func = np.hstack

    count = 0

    for idx, chunk in enumerate(db.iterator):
        # print('Retrieving data elements, chunk: ', idx)
        chunk = np.array(chunk.detach())

        # location = max(1, idx * db.batch_size)

        # lower_bound = index[(index / location) >= 1]
        # upper_bound = index[(index / location) < 2]

        # inds = np.intersect1d(lower_bound, upper_bound)
        # inds_to_get = inds % location

        # out = np.hstack((out, chunk[inds_to_get, col]))
        s_idx = count
        e_idx = chunk.shape[0] + s_idx
        out[s_idx:e_idx] = chunk[:, col]

        count = e_idx

    return out