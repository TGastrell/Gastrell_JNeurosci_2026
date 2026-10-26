import mne
import numpy as np
import pandas as pd
import pathlib

def load_epochs(sub_dir, t_min, t_max, bl_min, bl_max, bpf_low, bpf_high, eogs = False):
    """Load epochs data for a subject."""

    epoch_dir = sub_dir / 'eeg'
    file_stem = f'task-TG-MotEst-2.0_t-{t_min}-{t_max}_bl-{bl_min}-{bl_max}-{bpf_low}-{bpf_high}bpf-clean_epo.fif.gz'
    
    epochs_01 = mne.read_epochs(epoch_dir / 
                                f'{sub_dir.stem}_ses-MAIN-01_{file_stem}', 
                                preload=True)
    epochs_01.baseline = None
    epochs_02 = mne.read_epochs(epoch_dir / 
                                f'{sub_dir.stem}_ses-MAIN-02_{file_stem}', 
                                preload=True)
    epochs_02.baseline = None
    epochs = mne.concatenate_epochs([epochs_01, epochs_02]).pick_types(eeg=True, eog=eogs)
    epochs = epochs["abort == 'NO' & responseTime > 0.2"]
    return epochs

def load_all_epochs(sorted_pts, 
                    t_min, 
                    t_max, 
                    bl_min, 
                    bl_max, 
                    bpf_low, 
                    bpf_high, 
                    eogs = False, 
                    crop = (-0.25, 2.0), 
                    filter_query = None,
                    electrode_subset = None):
    '''
    Read in all epochs for each participant. Crop to the desired timepoints. Convert to DataFrame.
    '''
    all_data = {}
    all_metadata = {}
    electrodes = None
    
    for sub_dir in sorted_pts:
        if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
            continue    

        sub = sub_dir.stem
        print(f'loading epochs for {sub}')

        epochs = load_epochs(sub_dir, 
                t_min, 
                t_max, 
                bl_min, 
                bl_max,
                bpf_low,
                bpf_high,
                eogs = eogs
                )
        
        epochs.crop(*crop)
        
        if electrode_subset is not None:
            epochs = epochs.pick_channels(electrode_subset)

        if filter_query is not None:
            epochs = epochs[filter_query]

        sub_data = epochs.to_data_frame()
        electrodes = epochs.ch_names
        sub_data[electrodes] = sub_data[electrodes].astype('float32')

        all_metadata[sub] = epochs.metadata
        all_data[sub] = sub_data

    return pd.concat(all_data, axis=0, ignore_index=True), pd.concat(all_metadata, axis=0, ignore_index=True), electrodes

def load_all_epochs_with_metadata(sorted_pts, 
                    t_min, 
                    t_max, 
                    bl_min, 
                    bl_max, 
                    bpf_low, 
                    bpf_high, 
                    eogs = False, 
                    crop = (-0.25, 2.0), 
                    filter_query = None,
                    electrode_subset = None):
    '''
    Read in all epochs for each participant. Crop to the desired timepoints. Convert to DataFrame.
    '''
    all_data = {}
    all_metadata = {}
    electrodes = None
    
    for sub_dir in sorted_pts:
        if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
            continue    

        sub = sub_dir.stem
        print(f'loading epochs for {sub}')

        epochs = load_epochs(sub_dir, 
                t_min, 
                t_max, 
                bl_min, 
                bl_max,
                bpf_low,
                bpf_high,
                eogs = eogs
                )
        
        epochs.crop(*crop)
        
        if electrode_subset is not None:
            epochs = epochs.pick_channels(electrode_subset)

        if filter_query is not None:
            epochs = epochs[filter_query]

        sub_data = epochs.to_data_frame()
        electrodes = epochs.ch_names
        sub_data[electrodes] = sub_data[electrodes].astype('float32')

        #metadata = epochs.metadata
        #metadata['uniqueTrialID'] = metadata.cumcount()
        all_metadata[sub] = epochs.metadata
        all_data[sub] = sub_data

        # merge sub_data with metadata
        all_data[sub] = pd.merge(sub_data, epochs.metadata, left_on='epoch', right_index=True)

    return pd.concat(all_data, axis=0, ignore_index=True), electrodes

def get_all_evoked(sorted_pts, 
                   t_min, t_max, 
                   bl_min, bl_max, 
                   bpf_low, bpf_high, 
                   eogs = False, 
                   electrode_subset = None,
                   surface_laplacian = False,
                    crop = (-0.25, 2.0),
                    conditions = {'pk': [0, 5, 10], 'sk': [0.2, 0.4, 2]}):
    '''
    Read in all epochs for each participant. Return evoked data as dataframe.
    '''
    all_evokeds = {}
    all_metadata = {}
    for sub_dir in sorted_pts:
        if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
            continue    

        sub = sub_dir.stem
        print(f'loading epochs for {sub}')

        epochs = load_epochs(sub_dir, 
                t_min, 
                t_max, 
                bl_min, 
                bl_max,
                bpf_low,
                bpf_high,
                eogs = eogs
                )
        
        if electrode_subset is not None:
            epochs = epochs.pick_channels(electrode_subset)
        if crop is not None:
            epochs.crop(*crop)
        if surface_laplacian:
            epochs = mne.preprocessing.compute_current_source_density(epochs)
        sub_evokeds = {}
        for pk_idx, prior_kappa in enumerate(conditions['pk']):
            for sk_idx, stim_kappa in enumerate(conditions['sk']):
                query = f'priorKappa == {prior_kappa} & stimKappa == {stim_kappa}'
                evoked = epochs[query].average().to_data_frame()

                evoked['subjectNumber'] = sub
                evoked['priorKappa'] = prior_kappa
                evoked['stimKappa'] = stim_kappa
                sub_evokeds[(prior_kappa, stim_kappa)] = evoked
        evoked = pd.concat(sub_evokeds, axis=0, ignore_index=True)

        all_evokeds[sub] = evoked
        all_metadata[sub] = epochs.metadata

    return all_evokeds, all_metadata, epochs.ch_names

def get_all_gfp(sorted_pts, 

                   t_min, t_max, 
                   bl_min, bl_max, 
                   bpf_low, bpf_high, 
                   eogs = False, 
                   surface_laplacian = False,
                    crop = (-0.25, 2.0),
                    conditions = {'pk': [0, 5, 10], 'sk': [0.2, 0.4, 2]}):
    '''
    Read in all epochs for each participant. Return evoked data as dataframe.
    '''
    all_evokeds = {}
    all_metadata = {}
    for sub_dir in sorted_pts:
        if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
            continue    

        sub = sub_dir.stem
        print(f'loading epochs for {sub}')

        epochs = load_epochs(sub_dir, 
                t_min, 
                t_max, 
                bl_min, 
                bl_max,
                bpf_low,
                bpf_high,
                eogs = eogs
                )
        
        if crop is not None:
            epochs.crop(*crop)
        if surface_laplacian:
            epochs = mne.preprocessing.compute_current_source_density(epochs)
        sub_evokeds = {}
        for pk_idx, prior_kappa in enumerate(conditions['pk']):
            for sk_idx, stim_kappa in enumerate(conditions['sk']):
                query = f'priorKappa == {prior_kappa} & stimKappa == {stim_kappa}'
                e_df = epochs[query].to_data_frame()

                e_df['gfp'] = e_df[epochs.ch_names].std(axis=1)


                e_df_sum = (e_df[['time', 'gfp']]
                           .groupby(['time'])['gfp']
                           .mean()
                           .reset_index()
                           )
                
                e_df_sum['subjectNumber'] = sub
                e_df_sum['priorKappa'] = prior_kappa
                e_df_sum['stimKappa'] = stim_kappa
                sub_evokeds[(prior_kappa, stim_kappa)] = e_df_sum
        evoked = pd.concat(sub_evokeds, axis=0, ignore_index=True)

        all_evokeds[sub] = evoked
        all_metadata[sub] = epochs.metadata

    return all_evokeds, all_metadata, epochs.ch_names

