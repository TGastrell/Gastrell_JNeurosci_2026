"""
Docstring for FE_chans2metrics
Date: June 2025
Author: Tim Gastrell
Last updated: Jan 2026 (Documentation only)

This script calculates decoding metrics from predicted channel activations
stored in an MNE Epochs object. It computes decoded angles, decoding errors,
accuracy, vector lengths, biases, and tuning strengths for each trial and time point.

These metrics converted into a long-format DataFrame and saved as a parquet file for each subject, bundled with the behvaioural metadata.
This step makes it easy to link decoding results with behavioural data for further analysis.
"""
#%%
import argparse

import mne
import numpy as np
import pandas as pd
from analysis.common import utils
from analysis.common.epoch_config import BASELINE_WINDOWS, clean_epoch_label
from analysis.common.paths import WITHIN_PRIOR_FE_DIRECTORY, subject_directories


ANALYSIS_INPUTS = {
    "main": {
        "subdirectory": "train_sk2_unif",
        "stimulus_label": "stimDir",
    },
    "within-prior": {
        "subdirectory": WITHIN_PRIOR_FE_DIRECTORY,
        "stimulus_label": "binLabel",
    },
}


def get_metrics(epochs, dv='stimDir'):
    '''
    Calculate decoding metrics from predicted channel activations in epochs object.
    Inputs:
        epochs: MNE epochs object containing predicted channel activations.
        dv: string, name of the column in epochs.metadata representing the dependent variable (e.g., 'stimDir').
    
    Outputs:
        decoded_angle: n_time x n_trials matrix of decoded angles.
        decoded_error: n_time x n_trials matrix of decoding errors (decoded angle - dv).
        accuracy: n_time x n_trials matrix of decoding accuracy (cosine of decoded error).
        vector_length: n_time x n_trials matrix of vector lengths (magnitude of decoded angle).
        bias: n_time x n_trials matrix of biases (decoded error adjusted for prior direction).
        tuning_str: n_time x n_trials matrix of tuning strengths (vector length * accuracy).
        time_by_trial: n_time x n_trials matrix of time indices for each trial.
        trial_by_time: n_time x n_trials matrix of trial indices for each time point.
    '''
    data = epochs.get_data()

    metrics = {}
    
    dv = np.array(epochs.metadata.reset_index(drop = True)[dv])
    stims = np.array(epochs.metadata.reset_index(drop = True)['stimDir'])
    resp = np.array(epochs.metadata.reset_index(drop = True)['responseAngle'])
    priors = np.array(epochs.metadata.reset_index(drop = True)['priorDir'])
    unique_trial_index = epochs.metadata.reset_index(drop = True)['unique_trial_index']

    n_trials, n_chan, n_time = data.shape

    neuron_prefs = np.linspace(-np.pi, np.pi, n_chan, endpoint=False)



    theta = np.array([np.inner(data[:,:,t], np.exp(neuron_prefs * 1j)) for t in range(n_time)]) # Compute the complex vector sum of the channel responses
    decoded_angle = np.angle(theta)
    vector_length = np.abs(theta)

    decoded_error = np.array([utils.circular_difference(decoded_angle[t], dv) for t in range(n_time)])
    accuracy = np.cos(decoded_error) # Decode the error in the presented motion direction

    flip_msk = utils.circular_difference(priors, stims) < 0 #is prior rotated CW from stim?
    bias = decoded_error.copy()
    bias[:,flip_msk] *= -1 # Flip the sign of the error if the prior is rotated CW from the stimulus
    
    tuning_str = vector_length * accuracy # Compute the tuning strength. Cosine of the error penalises peaks in the wrong direction
    
    time_by_trial = np.array([unique_trial_index.to_list()] * n_time)
    trial_by_time = np.array([np.arange(n_time)] * n_trials).T

    return decoded_angle, decoded_error, accuracy, vector_length, bias, tuning_str, time_by_trial, trial_by_time # each is an n_time x n_trial matrix


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate decoding metrics from forward-encoding predictions."
    )
    parser.add_argument(
        "--analysis",
        required=True,
        choices=ANALYSIS_INPUTS,
        help="Select main or within-prior decoding predictions.",
    )
    parser.add_argument(
        "--baseline",
        choices=BASELINE_WINDOWS,
        default="standard",
        help="Select the prediction epoch baseline (default: standard, -0.35 to -0.25 s).",
    )
    parser.add_argument(
        "--time-window",
        choices=("cropped", "full"),
        default="cropped",
        help="Read predictions over -0.25..2 s (default) or the full -0.95..2 s.",
    )
    parser.add_argument(
        "--subject",
        help="Process one participant directory, for example sub-001 (default: all participants).",
    )
    args = parser.parse_args()

    analysis = ANALYSIS_INPUTS[args.analysis]
    t_min = -0.25 if args.time_window == "cropped" else -0.95
    file_label = clean_epoch_label(t_min, 2, args.baseline)

    for sub_dir in subject_directories(args.subject):
        sub = sub_dir.stem

        FE_dir = sub_dir / 'eeg' / 'FE' / analysis["subdirectory"]

        # load predicted channel activations
        epochs_file = FE_dir / f'{sub}_task-TG-MotEst-2.0_{file_label}_FE-pred-epochs.fif.gz'

        if not epochs_file.exists():
            if args.subject is not None:
                raise FileNotFoundError(f"Prediction epochs do not exist: {epochs_file}")
            print(f'File not found: {epochs_file}; skipping {sub}.')
            continue
        
        epochs = mne.read_epochs(epochs_file, preload=True)

        epochs.metadata['psd'] = np.abs(utils.circular_difference(epochs.metadata.reset_index(drop = True)['priorDir'], 
                                                            epochs.metadata.reset_index(drop = True)['stimDir']))
        epochs = epochs['abort == "NO" & responseTime > 0.2']

        n_trials, n_chans, n_time = epochs.get_data().shape

        epochs.metadata['unique_trial_index'] = np.arange(n_trials)

        metrics = (np.array(get_metrics(epochs, analysis["stimulus_label"])) # metrics x n_time x n_trials
                .T                            # t_trials x n_time x metrics
                .reshape(n_trials*n_time, 8)) # Final shape is trials*timepoints x metrics. the last columns corresponds to trial number and time point

        metric_df = pd.DataFrame(metrics, columns=['decoded_angle', 
                                                'decoded_error', 
                                                'accuracy', 
                                                'vector_length', 
                                                'bias', 
                                                'tuning_str',
                                                'unique_trial_index',
                                                'time'])

        sub_df = pd.merge(left = epochs.metadata, left_on='unique_trial_index', right = metric_df, right_on='unique_trial_index')
        out_pth = FE_dir / 'decoding_metrics'
        if not out_pth.exists():
            out_pth.mkdir(exist_ok=True, parents=True)

        output_fname = f'{sub}_task-TG-MotEst-2.0_{file_label}_FE-metrics-long.parquet'
        sub_df.to_parquet(out_pth / output_fname,
                    index= False)
# %%
