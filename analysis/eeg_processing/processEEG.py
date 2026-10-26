''' 
Date: June 2025
Author: Tim Gastrell
Last Edited: Jan 2026 (minor naming edits, documentation)

Custom EEG processing script using MNE and MNE-Faster.py to preprocess EEG data:
- Load BDF files and rename peripheral channels.
- Set montage and drop unnecessary channels.
- Apply band-pass and notch filtering.
- Epoch the data around specific events with baseline correction.
- Integrate behavioral data as metadata.

Designed to be run from the command line with input (rawdata/sub-xxx/eeg) and output (derivatives/sub-xxx/eeg) directories as arguments. 
This is useful for batch processing multiple subjects/sessions. 
I typically run this script on a high-performance computing cluster.

Major resource used: https://github.com/wmvanvliet/mne-faster/blob/main/examples/plot_artifact_correction_with_faster.py
'''
import mne 
import pandas as pd
import pathlib
import numpy as np
import argparse
from mne_faster import (find_bad_channels, find_bad_epochs,
                        find_bad_components, find_bad_channels_in_epochs)
from analysis.common.epoch_config import BASELINE_WINDOWS, epoch_label

def process_eeg(eeg_file, beh_file, output_dir, baseline="standard"):

    sub = eeg_file.stem.split('_')[0]
    ses = eeg_file.stem.split('_')[1]
    
    peripheral_channels = dict(EXG3 = "LH",
                            EXG4 = "RH",
                            EXG5 = "LV",
                            EXG6 = "UV",
                            EXG7 = "ML",
                            EXG8 = "MR")


    raw = mne.io.read_raw_bdf(eeg_file, preload=True, infer_types= True)
    raw.rename_channels(peripheral_channels)
    raw.set_channel_types({'LH': 'eog', 'RH': 'eog', 'LV': 'eog', 'UV': 'eog', 'ML': 'eog', 'MR': 'eog'})
    raw.drop_channels(['EXG1', 'EXG2'])

    raw.set_montage('standard_1020')
    raw.info['bads'] = []
    events = mne.find_events(raw, stim_channel='Status')

    # Special cases where trigs either fired or misfired due to technical issues.
    # These events should be removed from the data.
    # See readme files in rawdata for each subject for more details.
    match sub :
        case 'sub-016':
            bad_events = [4494035, 4494667, 4495376]
            events = events[~np.isin(events[:,0], bad_events)]
        case 'sub-027':
            bad_events = [2085588,2086202, 2086714]
            events = events[~np.isin(events[:,0], bad_events)]

    
    raw, _ = mne.set_eeg_reference(raw,ref_channels = 'average')

    picks = mne.pick_types(raw.info, eeg=True, eog=True, stim=False)

    bpf_low = 0.2 # band-pass filter low cut-off frequency
    bpf_high = 40 # band-pass filter high cut-off frequency
    tmin = -0.95 # epoch start time
    tmax = 2 # epoch end time
    bl_min, bl_max = BASELINE_WINDOWS[baseline]

    # Band-pass filter the data
    raw.filter(bpf_low, bpf_high, method='iir', picks=picks)
    raw.notch_filter(50, picks=picks)

    # Construct epochs. Note that we also include EOG channels.
    signal_events = mne.pick_events(events, include=[3])

    epochs = mne.Epochs(raw, 
                        signal_events, 
                        {'SIGNAL' : 3}, 
                        tmin, 
                        tmax, 
                        baseline= (bl_min, bl_max),
                        preload=True, 
                        picks=picks, 
                        detrend=None)
    
    # downsample to 256 Hz
    epochs.resample(256)

    # Add Behavioral data as Metadata
    bhv = pd.read_csv(beh_file, sep='\t').query("not dialStartPos.isnull()", engine = 'python')
    bhv.priorKappa = round(bhv.priorKappa * 2) /2

    epochs.metadata = bhv

    # Save uncleaned epochs for reference
    epochs_before = epochs.copy()
    epochs_before.set_eeg_reference('average')

    label = epoch_label(tmin, tmax, baseline, bpf_low, bpf_high)

    epochs_before.save(output_dir / rf'{sub}_{ses}_task-TG-MotEst-2.0_{label}-unclean_epo.fif.gz', overwrite=True)

    # Clean the data using FASTER

    # Step 1: mark bad channels
    epochs.info['bads'] = find_bad_channels(epochs, eeg_ref_corr=False)
    if len(epochs.info['bads']) > 0:
        epochs.interpolate_bads()

    # Step 2: mark bad epochs
    bad_epochs = find_bad_epochs(epochs)
    if len(bad_epochs) > 0:
        epochs.drop(bad_epochs)

    # Step 3: mark bad ICA components (using the build-in MNE functionality for this)
    ica = mne.preprocessing.ICA(0.99).fit(epochs)
    ica.exclude = find_bad_components(ica, epochs)
    ica.apply(epochs)
    # Need to re-baseline data after ICA transformation
    epochs.apply_baseline(epochs.baseline)

    # Step 4: mark bad channels for each epoch and interpolate them.
    bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, eeg_ref_corr=False)
    for i, b in enumerate(bad_channels_per_epoch):
        if len(b) > 0:
            ep = epochs[i]
            ep.info['bads'] = b
            ep.interpolate_bads() 
            epochs._data[i, :, :] = ep._data[0, :, :]


    epochs.set_eeg_reference('average')

    epochs.save(output_dir / rf'{sub}_{ses}_task-TG-MotEst-2.0_{label}-clean_epo.fif.gz', overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess EEG data for one participant.")
    parser.add_argument("input_dir", type=pathlib.Path, help="Participant folder under rawdata.")
    parser.add_argument("output_root", type=pathlib.Path, help="Derivatives output root.")
    parser.add_argument(
        "--baseline",
        choices=BASELINE_WINDOWS,
        default="standard",
        help="Epoch baseline window (default: standard, -0.35 to -0.25 s).",
    )
    args = parser.parse_args()
    input_dir = args.input_dir # / sub-xxx

    eeg_dir = input_dir / 'eeg'
    beh_dir = input_dir / 'beh'

    output_dir = args.output_root / eeg_dir.parts[-2] / eeg_dir.parts[-1] # output_root / sub-xxx / eeg

    if not output_dir.exists():
        output_dir.mkdir(parents=True,exist_ok= True)
    

    for eeg_file in eeg_dir.iterdir():
        # find relevant behavioural file path
        sub = eeg_file.stem.split('_')[0]
        ses = eeg_file.stem.split('_')[1]
        beh_file = beh_dir / f'{sub}_{ses}_task-TG-MotEst-2.0_beh.tsv'

        if not eeg_file.exists():
            raise ValueError(f'{eeg_file} is not a valid eeg file')
        if not beh_file.exists():
            raise ValueError(f'{beh_file} is not a valid behavioural file')
        
        process_eeg(eeg_file, beh_file, output_dir, baseline=args.baseline)