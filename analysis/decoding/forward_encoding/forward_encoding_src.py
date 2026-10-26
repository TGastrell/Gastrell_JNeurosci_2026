"""
Date: 2024-06
Author: Tim Gastrell
Last updated: Jan 2026 (Documentation only)

# Weights and predictions for forward encoding models
# In the forward encoding model, we estimate the weights that map sensor activations to the idealised channel responses. 
# We can then use these weights to predict the channel responses from the sensor activations. 
# The following functions are used to estimate the weights and predict the channel responses from the sensor activations.

"""
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf

# Set environment variables to limit the number of threads used by numpy and scipy
# This is useful to avoid issues with parallel processing and memory usage on HPCs with many cores.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import numpy as np

def process_loocv_trial(i, train, forward_model, noise_cov=False):
    """
    Process a single leave-one-out trial.
    Return predicted channel activations, or alternatively only the spatial filter
    Args:
        i (int): Index of the trial to leave out.
        train (np.ndarray): Training set of sensor activations (n_electrodes x n_trials).
        forward_model (np.ndarray): Idealised channel responses (n_channels x n_trials).
        noise_cov (bool): Whether to use noise covariance in weight estimation. Default is False.
        get_filter_only (bool): Whether to return only the spatial filter. Default is False.
    Returns:
        np.ndarray: Predicted channel activations (n_channels,) or spatial filter (n_channels x n_electrodes).
    """
    np.seterr(all='raise')
    # Leave-one-out split
    trn, tst = np.delete(train, i, axis=1), train[:, i]  
    c = np.delete(forward_model, i, axis=1)

    if noise_cov: # Use noise covariance in weight estimation. This is more robust but computationally more expensive.
        v = forward_encoding_train_weights_noise_cov(trn, c)

        pred = v@tst

        return pred, v
    else:
        # idealised channel responses per trial (computed above)
        c = np.delete(forward_model, i, axis=1)
        
        # compute weights from training data mapping sensor activations to the idealised channel response
        w = trn @ c.T @ np.linalg.pinv(c@c.T)
        
        # invert the model to predict channel responses from test sensor activations
        return np.linalg.pinv(w.T@w) @ w.T @ tst

def forward_encoding_loocv_parallel(train, forward_model, n_jobs=-1, noise_cov=False):
    """
    Perform leave-one-out cross-validation (LOOCV) in parallel over trials. One thread per trial.
    Args:
        train (np.ndarray): Training set of sensor activations (n_electrodes x n_trials).
        forward_model (np.ndarray): Idealised channel responses (n_channels x n_trials).
        n_jobs (int): Number of parallel jobs. Default is -1 (use all available cores).
        noise_cov (bool): Whether to use noise covariance in weight estimation. Default is False.
    Returns:
        np.ndarray: Predicted channel activations (n_trials x n_channels).
    """
    results = Parallel(n_jobs=n_jobs, backend='loky')(
                        delayed(process_loocv_trial)(i, train, forward_model, noise_cov) for i in range(train.shape[1])
                    )

    # Extract predictions and spatial filters if noise_cov is True
    if noise_cov:
        pred = np.array([res[0] for res in results])  # Predicted channel activations
        filters = np.array([res[1] for res in results])  # Spatial filters
        return pred, filters
    else:         
        pred = np.array(results)  # Predicted channel activations
    return pred


def forward_encoding_loocv(train, forward_model):
    '''
    train: training set of sensor activations (n_electrodes x n_trials)
    forward_model: idealised channel responses (n_channels x n_trials)
    '''
    #w = np.random.rand(train.shape[0], forward_model.shape[0])  # random weights that take ori chans to sensors
    pred = np.empty((train.shape[1], forward_model.shape[0]))

    # leave one out CV
    for i in range(train.shape[1]):
        # separate into training and testing 
        trn, tst = np.delete(train, i, axis=1), train[:, i]

        # idealised channel responses per trial (computed above)
        c = np.delete(forward_model, i, axis=1)
        
        # compute weights from training data mapping sensor activations to the idealised channel response
        w = trn @ c.T @ np.linalg.pinv(c@c.T)
        
        # invert the model to predict channel responses from test sensor activations
        pred[i] = np.linalg.pinv(w.T@w) @ w.T @ tst

    return pred


def forward_encoding_train_weights_noise_cov(trn, c):
    '''
    Train encoding weights for each channel and returns the optimal spatial filter given X and C. 
    This version of the code is optimised for numerical stability following issues with underflow on the HPC.

    trn: training set of sensor activations (n_electrodes x n_trials)
    c: idealised channel responses (n_channels x n_trials)
    '''
    v = np.empty((c.shape[0], trn.shape[0])) # Initialize weights array: n_channels x n_electrodes
    lw = LedoitWolf()
    
    # For each stimulus channel, compute the optimal spatial filter
    for n in range(c.shape[0]):
        # predicted activations for channel n. Shape: n_trials
        ci = c[n]

        # Compute weights via least squares

        denom = ci @ ci # demominator for weight computation. Scalar representing total power in channel n

        # Check for very small values that might risk underflow
        if np.abs(denom) < 1e-12:
            continue

        wi = (trn @ ci) / denom # trn @ ci represents the projection of sensor data onto the channel response. Normalise by total power in channel n (channel vector length) to get least squares model weights.

        # Check if weights are finite
        if not np.all(np.isfinite(wi)):
            continue

        # Compute covariance of residuals (noise covariance)
        # err = trn - wi[:, None] @ ci[:, None].T is equivalent, but the following is more numerically stable
        # Using outer product to avoid broadcasting issues
        err = trn - np.outer(wi, ci) # residuals: n_electrodes x n_trials. Outer product of weights and channel response gives predicted sensor activations for channel n.
        
        # Check if residuals are finite
        if not np.all(np.isfinite(err)):
            continue
        
        # Fit Ledoit-Wolf shrinkage model to estimate regularised noise covariance. This helps to stabilise the covariance estimate for inversion.
        lw.fit(err.T) # (trials x electrodes). 

        Sinv = np.linalg.pinv(lw.covariance_) # Inverse of regularised noise covariance matrix

        # Check covariances are finite
        if not np.all(np.isfinite(Sinv)):
            continue
        
        # Put it all together to get a spatial filter that optimally extracts channel n from the sensor data by accounting for corrleations in noise between sensors. 
        # This maximises the SNR of the channel estimates compared to ordinary least squares.

        numerator = Sinv @ wi # Numerator for the spatial filter (n_electrodes vector). Rotating the weights vector by the inverted noise covariance.
        denom2 = wi.T @ numerator # Denominator for the spatial filter. Accounts for the scaling of the weights after rotation by the noise covariance.

        # Check if denominator is too small or non-finite
        if np.abs(denom2) < 1e-12 or not np.isfinite(denom2):
            continue
        
        v[n] = numerator / denom2  # normalise the adjusted weights to obtain the optimal spatial filter for channel n  

    return v