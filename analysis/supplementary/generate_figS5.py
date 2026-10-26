#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.common import utils
from analysis.common.epoch_config import clean_epoch_label
from analysis.common.paths import (
    WITHIN_PRIOR_FE_DIRECTORY,
    analysis_cache_dir,
    derivatives_root,
    figure_output_path,
    within_prior_cache_dir,
)
from analysis.common.util_dataloader import load_decoding_data
from analysis.common.plotting_helpers import pairwise_cluster_tests, plot_timeseries_comparison, subject_level_summary

TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20
def generate_upper(axes):
    prior_colors = sns.color_palette("Oranges", n_colors=3)

    derivatives = derivatives_root()
    file_label = clean_epoch_label(-0.95, 2, "s5")
    pth = analysis_cache_dir() / f'trial_avg_decoding_metrics_{file_label}.parquet'

    if not pth.exists():
        print('Aggregated data not found, loading and processing individual subject data... \nThis only needs to be done once, subsequent runs will load the aggregated data and be much faster.')
        big_df= load_decoding_data(derivatives,
                            file_label,
                            ['subjectNumber',
                                            'trialNr',
                                            'time', 
                                            'stimKappa',
                                            'priorKappa',
                                            'stimDir', # swap for binLabel in binned analyses
                                            'priorDir',
                                            'responseAngle',
                                            'responseTime', 
                                            'decoded_angle', 
                                            'vector_length',
                                            'accuracy',
                                            'bias'],
                                            subdirectory='train_sk2_unif')

        big_df['uniqueTrialID'] = big_df.groupby(['subjectNumber']).cumcount()
        # Compute additional metrics
        big_df['prior_stim_dist'] = utils.circular_difference(big_df['priorDir'], big_df['stimDir'])
        big_df['signed_error'] = utils.get_signed_error(big_df['responseAngle'], big_df['prior_stim_dist'])
        big_df['decoding_error'] = utils.circular_difference(big_df['decoded_angle'], big_df['stimDir'])
        big_df['bias'] = utils.get_signed_error(big_df['decoding_error'], big_df['prior_stim_dist'])
        big_df['tuning_str'] = big_df['accuracy'] * big_df['vector_length']
        df = (big_df.groupby(['subjectNumber',
                        'time', 
                        'stimKappa',
                        'priorKappa'])
                        .agg({
                            'vector_length': 'mean',
                            'accuracy': 'mean',
                            'tuning_str': 'mean',
                            'bias': lambda x: utils.cmean(x),
                            'responseTime': 'mean'
                        })
                        .reset_index())
        df['precision'] = 1/ (big_df
                        .groupby(['subjectNumber', 'time', 'stimKappa', 'priorKappa'])['bias']
                        .agg(lambda x: utils.cstd(x))
                        .reset_index()['bias'])
        
        df.to_parquet(pth)
    else:
        print('Aggregated data found, loading...')
        df = pd.read_parquet(pth)

    print('Data loaded successfully. Applying final transformations...')
    df['time_secs'] = df['time'] / 256 - 0.95
    df['time_secs'] = df['time_secs'] - 0.7 # uncomment for early baseline analyses
    df = df.rename(columns={'stimKappa': 'Coherence', 
                            'priorKappa': 'Prior_Precision', 
                            'signed_error': 'Behavioural_Bias'})

    df['time'] = df['time_secs']

    def decorate_timeseries(ax):
        #ax.axvline(0.75, c='grey', linestyle='--')
        ax.axhline(0, c='grey', linestyle='--')
        ax.set(xlabel = 'Time (s)')

 ################################################################## PRIOR PRECISION
    ax = axes[0]

    # Aggregate data and run cluster tests for prior precision
    print('Aggregating data and running cluster tests for prior precision...')
    summary = subject_level_summary(df, cols = ['accuracy'], split_by = 'Prior_Precision', rt= True)
    cluster_results = pairwise_cluster_tests(summary, split_by='Prior_Precision')

    # convert cluster_results to long dataframe for easier plotting
    cluster_results_df = {}
    for comparison, (clusters, p_vals) in cluster_results.items():
        cluster_results_df[comparison] = pd.DataFrame({
            'time': df['time'].unique(),
            'T_obs': np.nan,
            'cluster': np.nan,
            'p_val': np.nan
        })
        for i_c, c in enumerate(clusters):
            c = c[0]
            cluster_results_df[comparison].loc[c, 'T_obs'] = np.nanmean(c)  # mean T value for the cluster
            cluster_results_df[comparison].loc[c, 'cluster'] = i_c
            cluster_results_df[comparison].loc[c, 'p_val'] = p_vals[i_c]

    cluster_results_pre = {}
    cluster_results_obs = {}
    cluster_results_resp = {}
    for comparison, df_cr in cluster_results_df.items():
        cluster_results_pre[comparison] = df_cr.query('time < 0')
        cluster_results_obs[comparison] = df_cr.query('time >= 0 & time < 0.75')
        cluster_results_resp[comparison] = df_cr.query('time >= 0.75')

    plot_timeseries_comparison(
                                summary = summary.query("time <= 0"),
                                cluster_results = cluster_results_pre, 
                            split_by = 'Prior_Precision',
                            ylim = (-.04, .12),
                            smooth = True,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    #ax.axvline(0, c='red')
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('a.u.', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)',fontsize=AX_LABEL_SIZE)
    ax.set_title('Pre-stimulus period', fontsize=PANEL_LABEL_SIZE)
    ##################################################################
    ax = axes[1]
    plot_timeseries_comparison(
                            summary = summary.query("time >= 0 & time <= 0.75"),
                            cluster_results = cluster_results_obs,
                            split_by = 'Prior_Precision', 
                            plot_rt = False,
                            smooth = True,
                            ylim = (-.04, .12),
                            legend = False,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.set_title('Observation period', fontsize=PANEL_LABEL_SIZE)

    ##################################################################
    ax = axes[2]
    plot_timeseries_comparison(
                            summary = summary.query("time >= 0.75"),
                            cluster_results = cluster_results_resp,
                            split_by = 'Prior_Precision', 
                            plot_rt = False,
                            smooth = True,
                            ylim = (-.04, .2),
                            legend = True,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.set_title('Response period', fontsize=PANEL_LABEL_SIZE)

def generate_lower(axes):
    prior_colors = sns.color_palette("Oranges", n_colors=3)

    derivatives = derivatives_root()
    file_label = clean_epoch_label(-0.95, 2, "s5")
    pth = within_prior_cache_dir() / f'trial_avg_decoding_metrics_{file_label}.parquet'

    if not pth.exists():
        print('Aggregated data not found, loading and processing individual subject data... \nThis only needs to be done once, subsequent runs will load the aggregated data and be much faster.')
        big_df= load_decoding_data(derivatives,
                            file_label,
                            ['subjectNumber',
                                            'trialNr',
                                            'time', 
                                            'stimKappa',
                                            'priorKappa',
                                            'binLabel',
                                            'priorDir',
                                            'responseAngle',
                                            'responseTime', 
                                            'decoded_angle', 
                                            'vector_length',
                                            'accuracy',
                                            'bias'],
                                            subdirectory=WITHIN_PRIOR_FE_DIRECTORY)
        big_df['stimDir'] = big_df['binLabel'] # uncomment for binned analyses
        big_df['uniqueTrialID'] = big_df.groupby(['subjectNumber']).cumcount()
        # Compute additional metrics
        big_df['prior_stim_dist'] = utils.circular_difference(big_df['priorDir'], big_df['stimDir'])
        big_df['signed_error'] = utils.get_signed_error(big_df['responseAngle'], big_df['prior_stim_dist'])
        big_df['decoding_error'] = utils.circular_difference(big_df['decoded_angle'], big_df['stimDir'])
        big_df['bias'] = utils.get_signed_error(big_df['decoding_error'], big_df['prior_stim_dist'])
        big_df['tuning_str'] = big_df['accuracy'] * big_df['vector_length']
        df = (big_df.groupby(['subjectNumber',
                        'time', 
                        'stimKappa',
                        'priorKappa'])
                        .agg({
                            'vector_length': 'mean',
                            'accuracy': 'mean',
                            'tuning_str': 'mean',
                            'bias': lambda x: utils.cmean(x),
                            'responseTime': 'mean'
                        })
                        .reset_index())
        df['precision'] = 1/ (big_df
                        .groupby(['subjectNumber', 'time', 'stimKappa', 'priorKappa'])['bias']
                        .agg(lambda x: utils.cstd(x))
                        .reset_index()['bias'])
        
        df.to_parquet(pth)
    else:
        print('Aggregated data found, loading...')
        df = pd.read_parquet(pth)

    print('Data loaded successfully. Applying final transformations...')
    df['time_secs'] = df['time'] / 256 - 0.95
    df['time_secs'] = df['time_secs'] - 0.7 # uncomment for early baseline analyses
    df = df.rename(columns={'stimKappa': 'Coherence', 
                            'priorKappa': 'Prior_Precision', 
                            'signed_error': 'Behavioural_Bias'})

    df['time'] = df['time_secs']

    def decorate_timeseries(ax):
        #ax.axvline(0.75, c='grey', linestyle='--')
        ax.axhline(0, c='grey', linestyle='--')
        ax.set(xlabel = 'Time (s)')

    ################################################################## PRIOR PRECISION
    ax = axes[0]

    # Aggregate data and run cluster tests for prior precision
    print('Aggregating data and running cluster tests for prior precision...')
    summary = subject_level_summary(df, cols = ['accuracy'], split_by = 'Prior_Precision', rt= True)
    cluster_results = pairwise_cluster_tests(summary, split_by='Prior_Precision')

    # convert cluster_results to long dataframe for easier plotting
    cluster_results_df = {}
    for comparison, (clusters, p_vals) in cluster_results.items():
        cluster_results_df[comparison] = pd.DataFrame({
            'time': df['time'].unique(),
            'T_obs': np.nan,
            'cluster': np.nan,
            'p_val': np.nan
        })
        for i_c, c in enumerate(clusters):
            c = c[0]
            cluster_results_df[comparison].loc[c, 'T_obs'] = np.nanmean(c)  # mean T value for the cluster
            cluster_results_df[comparison].loc[c, 'cluster'] = i_c
            cluster_results_df[comparison].loc[c, 'p_val'] = p_vals[i_c]

    cluster_results_pre = {}
    cluster_results_obs = {}
    cluster_results_resp = {}
    for comparison, df_cr in cluster_results_df.items():
        cluster_results_pre[comparison] = df_cr.query('time < 0')
        cluster_results_obs[comparison] = df_cr.query('time >= 0 & time < 0.75')
        cluster_results_resp[comparison] = df_cr.query('time >= 0.75')

    plot_timeseries_comparison(
                                summary = summary.query("time <= 0"),
                                cluster_results = cluster_results_pre, 
                            split_by = 'Prior_Precision',
                            ylim = (-.04, .12),
                            smooth = True,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    #ax.axvline(0, c='red')
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('a.u.', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    ##################################################################
    ax = axes[1]
    plot_timeseries_comparison(
                            summary = summary.query("time >= 0 & time <= 0.75"),
                            cluster_results = cluster_results_obs,
                            split_by = 'Prior_Precision', 
                            plot_rt = False,
                            smooth = True,
                            ylim = (-.04, .12),
                            legend = False,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    ##################################################################
    ax = axes[2]
    plot_timeseries_comparison(
                            summary = summary.query("time >= 0.75"),
                            cluster_results = cluster_results_resp,
                            split_by = 'Prior_Precision', 
                            plot_rt = False,
                            smooth = True,
                            ylim = (-.04, .2),
                            legend = False,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 3, figsize=(20,11))
    generate_upper(axes[0,:])
    generate_lower(axes[1,:])

    fig.text(0.01, 1, 'A', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.5, 'B', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    sns.despine()
    plt.tight_layout()
    plt.savefig(figure_output_path('figS5.svg'),
            dpi = 300,
            bbox_inches='tight')
    plt.show()

    # TODO:
    # Add Labels (A, B)
    # Maybe annotate the different decoders used?
# %%
