import numpy as np
import pandas as pd

from sklearn.datasets import make_regression


def generate_dataset_linear(n_samples=500, set_X=None):
    """
    Generate samples from two correlated distributions and allow for setting a specific
    value for X to simulate a RCT where patient's are assigned to experimental vs control groups.
    
    X is a bernoulli variable sampled from binomial distribution.
    Z is a confounder sampled from a uniform distribution.
    Y is a continuous real number. 

    The probabilistic model for this is:
    X --> Y
    Z --> X
    Z --> Y
    
    Args:
        n_samples (int): no. samples to generate
        set_X (arr): numpy array to set_X to a specific interventoin to simulate a RCT
                
    Returns:
        samples (pandas.DataFarme): a pandas dataframe of sampled data in the form [x, y, z]
    
    """
   
    z = np.random.uniform(size=n_samples)
    
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        p_x = np.minimum(np.maximum(z,0.1), 0.9)
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 = 2 * z
    y1 = y0 - 0.5

    y = np.where(x == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "z":z})


def generate_dataset_regression(n_features, rank, noise=0):
    """
    Generate a dataset of 100,000 samples with n_features features and of rank 'rank'.
    
    Args:
        n_features (int): no. of features to generate
        rank (int): rank of dataset
        noise (float): noise to add to the 
        
    Returns:
        dataset (pandas.Dataframe): a pandas Dataframe of regression data.
    
    """
    
    return make_regression(n_samples=100000, n_features=n_features, rank=rank, noise=noise)