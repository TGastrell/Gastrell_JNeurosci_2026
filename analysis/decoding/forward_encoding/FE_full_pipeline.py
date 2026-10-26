#%% 
# Author: Timothy Gastrell
# Forward encoding analysis pipeline for motion direction decoding from EEG data.
# This script trains a forward encoding model to decode motion direction from EEG data. 
# It includes functions for training the model, cross-decoding, and repackaging the predicted channel activations into a new MNE Epochs object. 
# The script is designed to be run on individual subject data, specified by a command-line argument.
# The trained model weights and predicted channel activations are saved for further analysis.

# I highly recommend running this script on a high-performance computing cluster, as the training and cross-validation steps can be computationally intensive, especially with the noise covariance estimation. 
# The script is set up to use parallel processing for the leave-one-out cross-validation step to speed up computation.

import numpy as np
import mne
import pathlib
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import argparse
from tqdm import tqdm
from analysis.common import eeg_utils
from analysis.common.epoch_config import BASELINE_WINDOWS, clean_epoch_label
from analysis.common.paths import (
    WITHIN_PRIOR_FE_DIRECTORY,
    subject_directories,
)
from analysis.decoding.forward_encoding.forward_encoding_src import (
    forward_encoding_train_weights_noise_cov,
    forward_encoding_loocv_parallel,
)

tuning_func = lambda x: np.maximum(0, np.cos(x)**5)

def get_forward_model(y, n_chan=16):
    """
    Generate the forward model (idealised channel responses) for given stimulus labels y.
    Args:
        y (np.ndarray): Array of shape (n_trials,) containing stimulus labels (motion directions in radians).
        n_chan (int): Number of basis channels.
    Returns:
        np.ndarray: Forward model of shape (n_chan, n_trials).
    """
    neuron_prefs = np.linspace(-np.pi, np.pi, n_chan, endpoint=False)
    # get idealised channel responses for each stimulus 
    return tuning_func(
        neuron_prefs[:,None]-y # difference between channel centers and presented motion directions
        )
    
def train_model(X, y):
    """
    Train the encoding model at each time point. Validate the model using leave-one-out cross-validation (LOOCV).
    Args:
        X (np.ndarray): EEG data of shape (n_trials, n_sensors, n_time).
        y (np.ndarray): Array of shape (n_trials,) containing stimulus labels (motion directions in radians).
    Returns:
        tuple: (weights, xval_pred, decoded_angle)
            - weights (np.ndarray): Trained weights of shape (n_chan, n_sensors, n_time).
            - xval_pred (np.ndarray): Predicted channel activations of shape (n_trials, n_chan, n_time).
            - decoded_angle (np.ndarray): Decoded angles of shape (n_time, n_trials).
            - xval_weights (np.ndarray): Spatial filters from LOOCV with noise covariance, of shape (n_trials, n_chan, n_sensors, n_time).
    """

    n_chan = 16
    n_trials = X.shape[0]
    n_elc = X.shape[1]
    n_time = X.shape[2]

    C = get_forward_model(y) # channels x trials

    # Initialize weights array
    weights = np.empty((n_chan, n_elc, n_time)) # channels x sensors x time

    # Demean X and C over trials
    X = X-X.mean(axis = (0))[None, :, :]
    C = C-C.mean(axis = (1))[:,None]

    # Initialize predicted channel activations array
    xval_pred = np.empty((n_trials, n_chan, n_time)) # trials x channels x time
    xval_weights = np.empty((n_trials, n_chan, n_elc, n_time)) # trials x channels x sensors x time this will store the spatial filters for each LOOCV iteration, if noise covariance is used
    
    # Train and validate at each time point
    for t in tqdm(range(n_time), 'Training...'):
        #logger.info(f'Training {t} of {n_time} timepoints')
        weights[...,t] = forward_encoding_train_weights_noise_cov(X[...,t].T, C) # Get weights for this timepoint. Sensors x channels

        #logger.info('Validating...')
        xval_pred[...,t], xval_weights[...,t] = forward_encoding_loocv_parallel(X[...,t].T, C, n_jobs=-1, noise_cov=True) # Get predicted channel activations for this timepoint. Trials x channels

    # Decode predicted angles from estimated channel activations. Dot product with complex-valued channel preferences.
    theta = np.array([np.inner(xval_pred[:,:,t], np.exp(np.linspace(-np.pi, np.pi, n_chan, endpoint=False) * 1j)) for t in range(n_time)])  # shape: n_time x n_trials
    decoded_angle = np.angle(theta)

    return weights, xval_pred, decoded_angle, xval_weights  # weights: channels x sensors x time, xval_pred: trials x channels x time, decoded_angle: n_time x n_trials

def xdecode(X, weights):
    """
    Cross decode the motion direction using a set of pre-trained weights.
    Args:
        X (np.ndarray): EEG data of shape (n_trials, n_sensors, n_time).
        weights (np.ndarray): Trained weights of shape (n_chan, n_sensors, n_time).
    Returns:
        tuple: (pred, decoded_angle)
            - pred (np.ndarray): Predicted channel activations of shape (n_trials, n_chan, n_time).
            - decoded_angle (np.ndarray): Decoded angles of shape (n_time, n_trials).
    """

    n_trials, n_elc, n_time = X.shape
    n_chan = weights.shape[0]

    # Demean X over trials
    X = X-X.mean(axis = (0))[None, :, :]

    # Initialize predicted channel activations array
    pred = np.empty((n_trials, n_chan, n_time)) # trials x channels x time

    # Decode at each time point
    for t in tqdm(range(n_time), 'Testing...'):

        #logger.info(f'Processing {t} of {n_time} timepoints')

        # Get the weights for this timepoint
        v = weights[...,t] # channels x sensors

        # Predict channel activations as the dot product of weights and sensor activations
        pred[:,:,t] = (v@X[..., t].T).T # trials x channels
    
    # Decode predicted angles from estimated channel activations. Dot product with complex-valued channel preferences.
    theta = np.array([np.inner(pred[:,:,t], np.exp(np.linspace(-np.pi, np.pi, n_chan, endpoint=False) * 1j)) for t in range(n_time)])  # shape: n_time x n_trials
    decoded_angle = np.angle(theta)

    return pred, decoded_angle  # pred: trials x channels x time, decoded_angle: n_time x n_trials

def repackage_epochs(xval_pred_chan, pred_chan, epochs, train_query, test_query):
    """
    Repackage the predicted channel activations into a new MNE Epochs object. This includes both training and test trials. 
    Trial order is not preserved, but is aligned with the metadata.
    Args:
        xval_pred_chan (np.ndarray): Predicted channel activations for training trials of shape (n_trials, n_chan, n_time).
        pred_chan (np.ndarray): Predicted channel activations for test trials of shape (n_trials, n_chan, n_time).
        epochs (mne.Epochs): Original epochs object containing all trials.
        train_query (str): Query string used to select training trials.
        test_query (str): Query string used to select test trials.
    Returns:
        mne.Epochs: New epochs object containing predicted channel activations for all trials.
    """

    all_pred = np.concatenate((xval_pred_chan, pred_chan), axis=0)  # concatenate training and test predictions
    all_metadata = pd.concat([epochs[train_query].metadata, epochs[test_query].metadata], ignore_index= True)   # metadata for all trials, need to reset index to concatenate
    all_events = np.concatenate((epochs[train_query].events, epochs[test_query].events), axis=0)  # events for all trials

    # Create a new epochs object with the 16 predicted channel activations
    ch_names = [f'pred_chan_{i}' for i in np.linspace(-np.pi, np.pi, pred_chan.shape[1], endpoint=False)]  # rename channels to 'pred_chan_1', 'pred_chan_2', ..., 'pred_chan_16'
    nchan = all_pred.shape[1]  # update the number of channels in the info
    sfreq = epochs.info['sfreq']  # keep the same sampling frequency
    new_info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=['misc'] * nchan,
        verbose=True
    ) 

    return mne.EpochsArray(all_pred,
                                info=new_info,
                                events=all_events,
                                event_id=epochs.event_id,
                                metadata=all_metadata.reset_index(drop=True))  # reset index to match metadata

def main(sub_dir, baseline="standard", time_window="cropped"):
    epoch_t_min = -0.95
    t_max = 2
    bl_min, bl_max = BASELINE_WINDOWS[baseline]
    bpf_low = 0.2
    bpf_high = 40

    if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
        raise ValueError(f"{sub_dir} is not a valid subject directory")

    sub = sub_dir.stem
    epochs = eeg_utils.load_epochs(sub_dir, 
                    epoch_t_min, 
                    t_max, 
                    bl_min, 
                    bl_max,
                    bpf_low,
                    bpf_high,
                    eogs = False
                    )
    
    if time_window == "cropped":
        epochs.crop(tmin=-0.25, tmax=t_max)
        prediction_t_min = -0.25
    elif time_window == "full":
        prediction_t_min = epoch_t_min
    else:
        raise ValueError(f"Unknown time window {time_window!r}; expected 'cropped' or 'full'.")
    
    # grab the training data
    train_query = f'priorKappa == 0 & stimKappa == 2' # training on unbiased and strong stimulus reliability trials
    X = epochs[train_query].get_data() # trials x elec x time
    y = epochs[train_query].metadata.reset_index(drop=True)['stimDir'].values  # presented motion directions

    # Train and validate the model
    (weights,           # channels x sensors x time weights mapping sensor acivtations to basis channel activations
    xval_pred_chan,    # trials x channels x time decoded basis channel activations
    xval_pred,          # n_time x n_trials decoded angles
    xval_weights        # trials x channels x sensors x time spatial filters from LOOCV with noise covariance. Used for temporal generalization
    ) = train_model(X, y)

    # Save the model weights
    output_dir = sub_dir / 'eeg' / 'FE' / 'train_sk2_unif'
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    label = clean_epoch_label(prediction_t_min, t_max, baseline, bpf_low, bpf_high)

    np.save(output_dir / f'{sub}_task-TG-MotEst-2.0_{label}_FE-weights.npy', weights) # channels x sensors x time.
    np.save(output_dir / f'{sub}_task-TG-MotEst-2.0_{label}_FE-xval-weights.npy', xval_weights) # trials x channels x sensors x time
    
    # Use pre-trained weights from above to predict the channel activations for each test trial
    test_query = f'not ({train_query})'
    X_test = epochs[test_query].get_data() # trials x elec x time

    (pred_chan, # trials x channels x time predicted channel activations
    pred # n_time x n_trials predicted angles
    ) = xdecode(X_test, weights)

    # Repackage the predicted channel activations into a new MNE Epochs object
    FE_epochs = repackage_epochs(xval_pred_chan, 
                                 pred_chan, 
                                 epochs, 
                                 train_query, 
                                 test_query)


    # Save the new epochs object with the predicted channel activations
    FE_epochs.save(output_dir / f'{sub}_task-TG-MotEst-2.0_{label}_FE-pred-epochs.fif.gz', overwrite = True)

def get_resampled_data(epochs):
    epochs.metadata['uniqueID'] = np.arange(0,len(epochs.metadata),1)
    df = epochs.metadata
    n_bins = 32
    bins = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    df['stimBin'] = np.digitize(df.stimDir, bins)
    df['binLabel'] = bins[df.stimBin.values -1] # bins in stim space

    def get_n(group):
        return np.min(np.histogram(group, n_bins)[0])

    df['n_to_sample'] = df.groupby(['priorKappa']).stimBin.transform(get_n)
    n_to_sample = df.n_to_sample.min() # enforce equal sampling for all conditions

    def resampler(group):
        return group.sample(n_to_sample, random_state=42)

    df_resamp = df.groupby(['priorKappa', 'stimBin']).apply(resampler).reset_index(drop=True)

    return epochs[df_resamp.uniqueID]

def within_prior(sub_dir, resamp=True, baseline="standard", time_window="cropped"):
    """Train/test the decoder separately within each prior condition."""
    t_min = -0.95
    t_max = 2
    bl_min, bl_max = BASELINE_WINDOWS[baseline]
    bpf_low = 0.2
    bpf_high = 40

    if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
        raise ValueError(f"{sub_dir} is not a valid subject directory")

    sub = sub_dir.stem
    epochs = eeg_utils.load_epochs(sub_dir, 
                    t_min, 
                    t_max, 
                    bl_min, 
                    bl_max,
                    bpf_low,
                    bpf_high,
                    eogs = False
                    )
    
    if time_window == "cropped":
        epochs.crop(tmin=-0.25, tmax=t_max)
        prediction_t_min = -0.25
    elif time_window == "full":
        prediction_t_min = t_min
    else:
        raise ValueError(f"Unknown time window {time_window!r}; expected 'cropped' or 'full'.")
    
    prior_conditions = [0,5,10]

    FE_condition_epochs = []

    output_dir = sub_dir / 'eeg' / 'FE' / WITHIN_PRIOR_FE_DIRECTORY

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    label = clean_epoch_label(prediction_t_min, t_max, baseline, bpf_low, bpf_high)
    if resamp:
        epochs = get_resampled_data(epochs)

    for condition in prior_conditions:
        logging.info(f'Running IEM on prior condition pk{condition}')
        condition_query = f'priorKappa == {condition}'
        condition_epochs = epochs[condition_query]

        # grab the training data
       # train_query = f'stimKappa != 2' # training on 
        X = condition_epochs.get_data() # trials x elec x time
        y = condition_epochs.metadata.reset_index(drop=True)['binLabel'].values  # presented motion directions

        # Train and validate the model
        (weights,           # channels x sensors x time weights mapping sensor acivtations to basis channel activations
        xval_pred_chan,    # trials x channels x time decoded basis channel activations
        xval_pred,          # n_time x n_trials decoded angles
        xval_weights        # trials x channels x sensors x time spatial filters from LOOCV with noise covariance. Used for temporal generalization
        ) = train_model(X, y)

        # Save the model weights


        np.save(output_dir / f'{sub}_task-TG-MotEst-2.0_{label}_pk{condition}_FE-weights.npy', weights) # channels x sensors x time.
        np.save(output_dir / f'{sub}_task-TG-MotEst-2.0_{label}_pk{condition}_FE-xval-weights.npy', xval_weights) # trials x channels x sensors x time
        

        # # Use pre-trained weights from above to predict the channel activations for each test trial
        # test_query = f'not ({train_query})'
        # X_test = condition_epochs[test_query].get_data() # trials x elec x time

        # (pred_chan, # trials x channels x time predicted channel activations
        # pred # n_time x n_trials predicted angles
        # ) = xdecode(X_test, weights)

        # # Repackage the predicted channel activations into a new MNE Epochs object
        # FE_condition_epochs.append(repackage_epochs(xval_pred_chan, 
        #                             pred_chan, 
        #                             condition_epochs, 
        #                             train_query, 
        #                             test_query))

        
        # Create a new epochs object with the 16 predicted channel activations
        ch_names = [f'pred_chan_{i}' for i in np.linspace(-np.pi, np.pi, xval_pred_chan.shape[1], endpoint=False)]  # rename channels to 'pred_chan_1', 'pred_chan_2', ..., 'pred_chan_16'
        nchan = xval_pred_chan.shape[1]  # update the number of channels in the info
        sfreq = epochs.info['sfreq']  # keep the same sampling frequency
        new_info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=['misc'] * nchan,
            verbose=True
        ) 
        FE_condition_epochs.append(mne.EpochsArray(xval_pred_chan,
                                info=new_info,
                                events=condition_epochs.events,
                                event_id=condition_epochs.event_id,
                                metadata=condition_epochs.metadata.reset_index(drop=True)))
        
    FE_epochs = mne.concatenate_epochs(FE_condition_epochs)
    # Save the new epochs object with the predicted channel activations
    FE_epochs.save(output_dir / f'{sub}_task-TG-MotEst-2.0_{label}_FE-pred-epochs.fif.gz', overwrite = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forward-encoding analysis.")
    parser.add_argument(
        "--analysis",
        choices=("main", "within-prior"),
        default="main",
        help="Run the main decoder or the within-prior decoder.",
    )
    parser.add_argument(
        "--baseline",
        choices=BASELINE_WINDOWS,
        default="standard",
        help="Select the epoch baseline (default: standard, -0.35 to -0.25 s).",
    )
    parser.add_argument(
        "--time-window",
        choices=("cropped", "full"),
        default="cropped",
        help="Predict over -0.25..2 s (default) or the full -0.95..2 s epochs.",
    )
    parser.add_argument(
        "--subject",
        help="Process one participant directory, for example sub-001 (default: all participants).",
    )
    args = parser.parse_args()
    for sub_dir in subject_directories(args.subject):
        if args.analysis == "main":
            main(
                sub_dir,
                baseline=args.baseline,
                time_window=args.time_window,
            )
        else:
            within_prior(
                sub_dir,
                baseline=args.baseline,
                time_window=args.time_window,
            )



# %%
