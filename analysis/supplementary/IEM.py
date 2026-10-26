
import numpy as np
from sklearn.covariance import LedoitWolf

def train_encoding_weights(c, X_trn, X_tst=None):
    '''
    Train encoding weights for each channel and returns predicted channel responses for test data. 
    This is a standard IEM implementation without noise covariance correction.
    Test data is optional - if not provided, the function will return the trained weights only.
    Use predict_channel_responses() to get predicted channel responses for test data after training if not done in one step.

    c: idealised channel responses - the forward model (n_channels x n_trials)
    X_trn: training set of sensor activations (n_electrodes x n_trials)
    X_tst: test set of sensor activations (n_electrodes x n_trials)

    Returns:
    w: encoding weights mapping sensor activations to channel responses (n_electrodes x n_channels)
    pred: predicted channel responses for test data (n_channels x n_trials)
    '''
    if X_trn.shape[1] != c.shape[1]:
        raise ValueError("Number of trials in X_trn and c must match.")

    # compute weights from training data mapping sensor activations to the idealised channel response
    w = X_trn @ c.T @ np.linalg.pinv(c@c.T)
    
    # invert the model to predict channel responses from test sensor activations
    if X_tst is not None:
        return w, predict_channel_responses(w, X_tst)
    else:
        return w # If no test data provided, return weights only

def predict_channel_responses(w, X_tst):
    '''
    Use trained encoding weights to predict channel responses for test data.
    
    w: encoding weights mapping sensor activations to channel responses (n_electrodes x n_channels)
    X_tst: test set of sensor activations (n_electrodes x n_trials)

    Returns:
    pred: predicted channel responses for test data (n_channels x n_trials)
    '''
    return np.linalg.pinv(w.T@w) @ w.T @ X_tst

def iem_noise_cov(c, X_trn, X_tst=None):
    v = opt_spatial_filter(c, X_trn)

    if X_tst is not None:
        pred = v@X_tst

        return v, pred
    else:
        return v

def get_forward_model(y, n_chan=16, endpoint=False):
    """
    Generate the forward model (idealised channel responses) for given stimulus labels y.
    Args:
        y (np.ndarray): Array of shape (n_trials,) containing stimulus labels (motion directions in radians).
        n_chan (int): Number of basis channels.
        endpoint (bool): Whether to include the endpoint in the tuning curve (i.e., whether the last channel is centered at 360 degrees or just before it).
    Returns:
        np.ndarray: Forward model of shape (n_chan, n_trials).
    """
    tuning_func = lambda x: np.maximum(0, np.cos(x)**5)

    neuron_prefs = np.linspace(-np.pi, np.pi, n_chan, endpoint=endpoint)
    # get idealised channel responses for each stimulus 
    return tuning_func(
        neuron_prefs[:,None]-y # difference between channel centers and presented motion directions
        )

def align_C_test(C_test, y, forward_model):
    '''
    Align decoded channel responses to the presented stimulus direction.

    Parameters:
    C_test: np.ndarray of shape (n_trials, n_channels, n_timepoints) containing decoded channel responses for each trial and timepoint
    y: np.ndarray of shape (n_trials,) containing stimulus directions in radians for each trial
    forward_model: np.ndarray of shape (n_channels, n_trials) containing idealised channel responses for each trial
    Returns:
    aligned_responses: np.ndarray of shape (n_trials, n_channels, n_timepoints) containing decoded channel responses circularly shifted to align with stimulus direction
    '''
    # align decoded responses to stimulus direction by circularly shifting each trial's channel responses so that the channel tuned to the presented stimulus direction is centered at 0 radians
    C_test_aligned = np.empty_like(C_test)
    for i in range(C_test.shape[0]):
        # compute how much to circularly shift based on stimulus direction
        shift = int(np.round(y[i] / (2*np.pi) # convert stimulus direction to a proportion of the full circle (0 to 1)
                                * forward_model.shape[0])) # convert proportion to number of channels to shift (rounded off to nearest integer)
        C_test_aligned[i] = np.roll(C_test[i], -shift, axis=0) # circularly shift decoded responses by the computed amount to align them to the presented stimulus direction
    return C_test_aligned

def opt_spatial_filter(c, trn):
    '''
    Train encoding weights for each channel and returns the optimal spatial filter given X and C. 
    This version of the code is optimised for numerical stability following issues with underflow on the HPC.

    trn: training set of sensor activations (n_electrodes x n_trials)
    c: idealised channel responses (n_channels x n_trials)

    Returns:
    v: optimal spatial filter mapping sensor activations to channel responses (n_channels x n_electrodes)
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