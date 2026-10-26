#%%
from pathlib import Path
import mne
from analysis.common import eeg_utils, tempgen_utils, utils
from analysis.common.paths import analysis_cache_dir, derivatives_root, figure_output_path
from scipy.stats import false_discovery_control
from analysis.common.util_dataloader import load_decoding_data
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from analysis.common.plotting_helpers import plot_tempGen, plot_single_timeseries
#%%
def generate_fig05():
    derivatives = derivatives_root()
    file_label='t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean'
    pth = analysis_cache_dir() / f'trial_avg_decoding_metrics_long_{file_label}.parquet'

    if not pth.exists():
        print('Aggregated data not found, loading and processing individual subject data... \nThis only needs to be done once, subsequent runs will load the aggregated data and be much faster.')
        big_df= load_decoding_data(derivatives,
                            file_label,
                            ['subjectNumber',
                                            'trialNr',
                                            'time', 
                                            'stimKappa',
                                            'priorKappa',
                                            'stimDir',
                                            'priorDir',
                                            'responseAngle',
                                            'responseTime', 
                                            'decoded_angle', 
                                            'vector_length',
                                            'accuracy',
                                            'bias'])
        big_df['uniqueTrialID'] = big_df.groupby(['subjectNumber']).cumcount()
        # Compute additional metrics
        big_df['prior_stim_dist'] = utils.circular_difference(big_df['priorDir'], big_df['stimDir'])
        big_df['signed_error'] = utils.get_signed_error(big_df['responseAngle'], big_df['prior_stim_dist'])
        big_df['decoding_error'] = utils.circular_difference(big_df['decoded_angle'], big_df['stimDir'])
        big_df['bias'] = utils.get_signed_error(big_df['decoding_error'], big_df['prior_stim_dist'])

        df = (big_df.groupby(['subjectNumber',
                        'time', 
                        'stimKappa',
                        'priorKappa'])
                        .agg({
                            'vector_length': 'mean',
                            'accuracy': 'mean',
                            'bias': lambda x: utils.cmean(x),
                            'responseTime': 'mean'
                        })
                        .reset_index())
        df['precision'] = 1/ (big_df
                        .groupby(['subjectNumber', 'time', 'stimKappa', 'priorKappa'])['bias']
                        .agg(lambda x: utils.cstd(x))
                        .reset_index()['bias'])
        
        df['time_secs'] = df['time'] / 256 - 0.25
        df.to_parquet(pth)
    else:
        print('Aggregated data found, loading...')
        df = pd.read_parquet(pth)

    print('Data loaded successfully. Applying final transformations...')
    df = df.query('time_secs >= -0.25')
    df = df.rename(columns={'stimKappa': 'Coherence', 
                            'priorKappa': 'Prior_Precision', 
                            'signed_error': 'Behavioural_Bias'})

    df['time'] = df['time_secs']

    # Sensor Topomaps of Regression p-values
    epo_pth = Path(derivatives /'sub-001')
    data_path = analysis_cache_dir() / 'sensor_reg_stats_all.pkl'

    t_min = -0.95
    t_max = 2
    bl_min = -0.35
    bl_max = -0.25
    bpf_low = 0.2
    bpf_high = 40

    # Preload all the data 
    print('Preloading data for regression and topomaps...')
    print('Loading an Epochs object...')
    epochs = eeg_utils.load_epochs(epo_pth, t_min, t_max, bl_min, bl_max, bpf_low, bpf_high, eogs = True).crop(-0.25, 2.0) # load an epochs object just to get the info for plotting topomaps. Cropping to the same time window as the decoding data for consistency.

    with open(data_path, 'rb') as f:
        print('Loading regression results, this is a large file so may take a while...')
        reg_dict = pickle.load(f)

    # Loading temporal generalisation matrices
    #derivatives = Path(r'..\..\derivatives')#
    print('Loading temporal generalisation matrices... Please wait, this may take a moment as the data is quite large.')
    tempGenMats = tempgen_utils.load_tempgen_matrices_hdf(derivatives)

    fig = plt.figure(figsize=(12,4))
    gs = GridSpec(2,6)

    ax = fig.add_subplot(gs[0, :])

    print('Plotting sensor topomaps...')
    p_val = false_discovery_control(reg_dict['p_val'], method='bh', axis = 1)
    p_val = false_discovery_control(p_val, method='bh', axis = 0)
    timepoints = [0.0, 0.4, 0.7, 1, 1.5, df['responseTime'].mean() + .85]
    sens_evk = mne.EvokedArray(-np.log(p_val),
                            info = epochs.info,
                            tmin = epochs.tmin).crop(-0.25, 2)
    sens_evk.set_montage('biosemi64')
    # Create boolean mask
    p_mask = p_val < 0.05
    sens_evk.plot_topomap(
        times = timepoints,
        ch_type = 'eeg',
        scalings = dict(eeg=1),
        time_format = '%0.1f s',
        vlim = (-5,8),
        show = False,
        colorbar = False,
        cmap = 'viridis',
        mask=p_mask,
        mask_params = dict(marker='^', markerfacecolor='white', markeredgecolor='black',
                        linewidth=0, markersize=10),
        axes = [fig.add_subplot(gs[0, i]) for i in range(6)],

    )
    fig.axes[0].set_axis_off()
    # After plotting
    for i, ax in enumerate(fig.axes[1:7]):  # Axes 1-6 are topomaps
        title = ax.get_title()
        ax.set_title(title, fontsize=16) 

    plt.savefig(figure_output_path('fig05a.svg'), dpi=300, bbox_inches='tight')
    fig = plt.figure(figsize=(12,12))
    gs = GridSpec(6,6)
    # Temporal Generalisation Matrices
    print('Plotting temporal generalisation matrix...')
    ax = fig.add_subplot(gs[3:,1:5])

    # Average over conditions
    mean_tempGenMats = np.mean(tempGenMats, axis=(1,2)) 

    plot_tempGen(mean_tempGenMats, ax=ax, 
                n_permutations=10000, # reduce for speed, should be 10000 for actual analysis
                rt = df['responseTime'].mean() + .85)

    print('Plotting decoding accuracy timeseries...')

    # Decoding Accuracy timeseries
    ax = fig.add_subplot(gs[1:3,:3])
    plot_single_timeseries(df.query("time <= 0.75")[['subjectNumber','time','accuracy']], 'accuracy', ax, 'gold', plot_clusters=True)
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.axvline(x=0, color='red')
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(x=0.75, color='grey', linestyle='--')
    ax.set_ylabel('a.u.', fontsize=16)
    ax.set_title('Observation Period', fontsize=16)
    ax.tick_params(labelsize=14)

    ax = fig.add_subplot(gs[1:3,3:])
    plot_single_timeseries(df.query("time >= 0.75")[['subjectNumber','time','accuracy', 'responseTime']], 'accuracy', ax, 'gold', plot_clusters=True, plot_rt=True)
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(x=0.75, color='grey', linestyle='--')
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_title('Response Period', fontsize=16)
    ax.tick_params(labelsize=14)


    fig.text(0.01, 1, 'A', fontsize=20, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.82, 'B', fontsize=20, fontweight='bold', va='top', ha='left')
    fig.text(0.16, 0.48, 'C', fontsize=20, fontweight='bold', va='top', ha='left')
    fig.tight_layout()

    plt.savefig(figure_output_path('fig05bc.svg'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    generate_fig05()
