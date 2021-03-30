import numpy as np
import pandas as pd

def logistic_fx(x):
    """
    Simplified logistic function from ML where L = 1 and k = 1, and x_0 = 0. 
    Maps a real number from (-inf, inf) to (0,1) centered on x=0.
    
    Args:
        x (float): input
    Returns:
        y (float): output
    """

    y = 1 / (1 + np.exp(-x))

    return y


def estimate_effect(df):
    """
    Estimated effect is the difference in response due to a treatment/intervention as measured in a 
    RCT or A/B test. We can simply calculate this as a difference in means with a
    95% CI between the responses of our control and experimental groups. 
    
    Args:
        df (pandas.DataFrame): a dataframe of samples.
        
    Returns:
        estimated_effect (dict[Str: float]): dictionary containing the difference in means for the
            treated and untreated samples and the "standard_error" - 90% confidence intervals arround "estimated_effect"    
    """

    base = df[df.x == 0]
    variant = df[df.x == 1]
    delta = variant.y.mean() - base.y.mean()
    delta_err = 1.96 * np.sqrt(variant.y.var() / variant.shape[0] + base.y.var() / base.shape[0])
    
    return {"estimated_effect": delta, "standard_error": delta_err}


def run_ab_test(datagenerator, n_samples=10000):
    """
    Generates n_samples from a datagenerator with the value of X randomized
    so that 50% of the samples recieve treatment X=1 and 50% receive X=0,
    and feeds the results into `estimate_effect` to get an unbiased 
    estimate of the average treatment effect.
    
    Args:
        datagenerator (method): a datagenerator method from datagen
        n_samples (int): an integer describing number of samples to draw
    Returns:
        estimated_effect (dict[Str: float]): See estimate_effect for details
    """
    n_samples_a = int(n_samples / 2)
    n_samples_b = n_samples - n_samples_a
    
    set_X = np.concatenate([np.ones(n_samples_a), np.zeros(n_samples_b)]).astype(np.int64)
    ds = datagenerator(n_samples=n_samples, set_X=set_X)

    return estimate_effect(ds)