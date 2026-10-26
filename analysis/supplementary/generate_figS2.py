#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from analysis.common import utils
from analysis.common.paths import (
    WITHIN_PRIOR_FE_DIRECTORY,
    derivatives_root,
    figure_output_path,
    within_prior_cache_dir,
)
from scipy.stats import false_discovery_control
from analysis.common.util_dataloader import load_decoding_data
import pickle
from analysis.common.plotting_helpers import pairwise_cluster_tests, plot_pvals, plot_timeseries_comparison, subject_level_summary

TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20

def generate_figS2():
    stim_colors = sns.color_palette("Blues", n_colors=3)
    prior_colors = sns.color_palette("Oranges", n_colors=3)

    derivatives = derivatives_root()
    file_label='t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean'
    pth = within_prior_cache_dir() / f'trial_avg_decoding_metrics_resamp_binned_long_{file_label}.parquet'

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
                                            'stimDir',
                                            'priorDir',
                                            'responseAngle',
                                            'responseTime', 
                                            'decoded_angle', 
                                            'vector_length',
                                            'accuracy',
                                            'bias'],
                                            subdirectory=WITHIN_PRIOR_FE_DIRECTORY)
        big_df['stimDir'] = big_df['binLabel']
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
    df = df.query('time_secs >= -0.95')
    df = df.rename(columns={'stimKappa': 'Coherence', 
                            'priorKappa': 'Prior_Precision', 
                            'signed_error': 'Behavioural_Bias'})

    df['time'] = df['time_secs']
  ################################################### Plot Decoding Accuracy split by stimulus and prior precision #####################################################

    def decorate_timeseries(ax):
        ax.axvline(0.75, c='grey', linestyle='--')
        ax.axhline(0, c='grey', linestyle='--')
        ax.set(xlabel = 'Time (s)')

    fig  = plt.figure(figsize=(16,13))
    gs = GridSpec(12,2)

    ## Decoding Accuracy timeseries by Coherence and Prior Precision
    ax = fig.add_subplot(gs[:5,0])

    # Aggregate data and run cluster tests for stimulus coherence
    print('Aggregating data and running cluster tests for stimulus coherence...')
    summary = subject_level_summary(df, cols = ['accuracy'], split_by = 'Coherence', rt= True)
    cluster_results = pairwise_cluster_tests(summary, split_by='Coherence')

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

    cluster_results_obs = {}
    cluster_results_resp = {}
    for comparison, df_cr in cluster_results_df.items():
        cluster_results_obs[comparison] = df_cr.query('time <= 0.75')
        cluster_results_resp[comparison] = df_cr.query('time >= 0.75')

    plot_timeseries_comparison(
                                summary = summary.query("time <= 0.75"),
                                cluster_results = cluster_results_obs, 
                            split_by = 'Coherence',
                            smooth = True, 
                            ylim = (-.04, .1),
                            ax = ax,
                            palette=stim_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.axvline(0, c='red')

    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_title('Observation Period', fontsize=PANEL_LABEL_SIZE)
    ax.set_ylabel('a.u.', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    ##################################################################
    ax = fig.add_subplot(gs[:5,1])
    plot_timeseries_comparison(
                                summary = summary.query("time >= 0.75"),
                                cluster_results = cluster_results_resp, 
                            split_by = 'Coherence',
                            plot_rt = True,
                            smooth = True, 
                            legend=True,
                            ax = ax,
                            palette=stim_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_title('Response Period', fontsize=PANEL_LABEL_SIZE)
    ax.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    ##################################################################
    ax = fig.add_subplot(gs[5:10,0])

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

    cluster_results_obs = {}
    cluster_results_resp = {}
    for comparison, df_cr in cluster_results_df.items():
        cluster_results_obs[comparison] = df_cr.query('time <= 0.75')
        cluster_results_resp[comparison] = df_cr.query('time >= 0.75')

    plot_timeseries_comparison(
                                summary = summary.query("time <= 0.75"),
                                cluster_results = cluster_results_obs, 
                            split_by = 'Prior_Precision',
                            ylim = (-.04, .1),
                            smooth = True,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    ax.axvline(0, c='red')
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('a.u.', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    ##################################################################
    ax = fig.add_subplot(gs[5:10,1])
    plot_timeseries_comparison(
                            summary = summary.query("time >= 0.75"),
                            cluster_results = cluster_results_resp,
                            split_by = 'Prior_Precision', 
                            plot_rt = True,
                            smooth = True,
                            legend = True,
                            ax = ax,
                            palette=prior_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ##################################################################

    # Load fitted model for plotting p-values of fixed effects
    print('Loading fitted model for plotting p-values of fixed effects...')
    mod_pth = within_prior_cache_dir()
    with open(mod_pth / 'acc_resamp_binned_model_fit_full_window.pkl', 'rb') as file:
        mod = pickle.load(file)


    ax = fig.add_subplot(gs[10:,0])

    # Apply FDR correction to the p-values of the fixed effects
    fdr_p = mod['pvals'].copy()
    fdr_p.iloc[:, :-2] = false_discovery_control(fdr_p.iloc[:, :-2], axis=0, method='bh')

    fdr_p = (fdr_p.rename(columns = {'Prior_Precision': rf'$\kappa_p$',
                                    'Coherence': rf'$\kappa_s$',
                                    'Coherence:Prior_Precision': rf'$\kappa_s$ x $\kappa_p$'})
                                    .drop(columns=['subjectNumber Var']))

    fdr_p['time'] = (fdr_p['time'] / 256) - 0.95
    plot_pvals(fdr_p.query('time <= 0.75'), 
            dv='Decoding Accuracy', 
            legend=False,
            intercept=False, 
            ax=ax)

    ax = fig.add_subplot(gs[10:,1])
    plot_pvals(fdr_p.query('time >= 0.75'), 
            dv='Decoding Accuracy', 
            legend=True, 
            intercept=False,
            ax=ax)

    sns.despine()

    # Finishing touches 
    fig.text(0.01, 1, 'A', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.6, 'B', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.18, 'C', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    plt.tight_layout()

    plt.savefig(figure_output_path('figS2.svg'),
                dpi=300, 
                bbox_inches='tight'
                )

if __name__ == "__main__":
    generate_figS2()
    plt.show()
# %%
