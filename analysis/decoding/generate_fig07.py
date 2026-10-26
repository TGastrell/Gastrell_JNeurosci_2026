#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.common import utils
from analysis.common.paths import analysis_cache_dir, derivatives_root, figure_output_path
from matplotlib.gridspec import GridSpec
from scipy.stats import false_discovery_control
import pickle
from pathlib import Path
from analysis.common.util_dataloader import load_decoding_data
from analysis.common.plotting_helpers import pairwise_cluster_tests, plot_pvals, plot_timeseries_comparison, subject_level_summary_circular

TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20

def generate_fig07():
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
        print('Loading aggregated data...')
        df = pd.read_parquet(pth)

    print('Data loaded successfully. Applying final transformations...')
    df = df.query('time_secs >= -0.25')
    df = df.rename(columns={'stimKappa': 'Coherence', 
                            'priorKappa': 'Prior_Precision', 
                            'signed_error': 'Behavioural_Bias'})

    df['time'] = df['time_secs']

    derivatives = derivatives_root()
    prior_colors = sns.color_palette('Oranges', n_colors=3)
    stim_colors = sns.color_palette('Blues', n_colors=3)

    
    def decorate_timeseries(ax):
        ax.axvline(0.75, c='grey', linestyle='--')
        ax.axhline(0, c='grey', linestyle='--')
        ax.set(xlabel = 'Time (s)')

    fig  = plt.figure(figsize=(16,13))  
    gs = GridSpec(12,2)

    
    ###################################################################################
    ## Decoding Bias timeseries by Coherence and Prior Precision
    ax = fig.add_subplot(gs[0:5,0])
    summary = subject_level_summary_circular(df, cols = ['bias'], split_by = 'Coherence')
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
        cluster_results_obs[comparison] = df_cr.query('time < 0.75')
        cluster_results_resp[comparison] = df_cr.query('time >= 0.75')

    plot_timeseries_comparison(
                                summary = summary.query('time < 0.75'),
                               #cluster_results = cluster_results_obs, 
                            split_by = 'Coherence',
                            smooth = True,
                            is_circular = True, 
                            ylim = (-1.5, 1.5),
                            ax = ax,
                            legend = False,
                            palette=stim_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.axvline(0, c='red')
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('Radians', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.axhspan(xmin=0, xmax=0.475, ymin=-1.5, ymax=1.5, color='grey', alpha=0.1)
    ax.set_title('Observation Period', fontsize=PANEL_LABEL_SIZE)

    ## Decoding Bias timeseries by Coherence and Prior Precision
    ax = fig.add_subplot(gs[0:5,1])

    plot_timeseries_comparison(
                                summary = summary.query('time >= 0.75'),
                                cluster_results = cluster_results_resp, 
                            split_by = 'Coherence',
                            smooth = True,
                            is_circular = True, 
                            ylim = (-1.5, 1.5),
                            ax = ax,
                            legend = True,
                            palette=stim_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.set_title('Response Period', fontsize=PANEL_LABEL_SIZE)

    ##################################################################
    ax = fig.add_subplot(gs[5:10,0])
    summary = subject_level_summary_circular(df, cols = ['bias'], split_by = 'Prior_Precision')
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
        cluster_results_obs[comparison] = df_cr.query('time < 0.75')
        cluster_results_resp[comparison] = df_cr.query('time >= 0.75')

    plot_timeseries_comparison(
                                summary = summary.query('time < 0.75'),
                                cluster_results = cluster_results_obs, 
                            split_by = 'Prior_Precision',
                            is_circular = True,
                            ylim = (-1.5, 1.5),
                            smooth = True,
                            ax = ax,
                            legend = False,
                            palette=prior_colors)

    # Customize plot
    ax.axvline(0, c='red')
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('Radians', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.axhspan(xmin=0, xmax=0.475, ymin=-1.5, ymax=1.5, color='grey', alpha=0.1)

    ax = fig.add_subplot(gs[5:10,1])

    plot_timeseries_comparison(
                                summary = summary.query('time >= 0.75'),
                               cluster_results = cluster_results_resp, 
                            split_by = 'Prior_Precision',
                            is_circular = True,
                            ylim = (-1.5, 1.5),
                            smooth = True,
                            ax = ax,
                            legend = True,
                            palette=prior_colors)

    # Customize plot
    decorate_timeseries(ax)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    ##################################################################

    # Load fitted model for plotting p-values of fixed effects
    print('Loading fitted model for plotting p-values of fixed effects...')
    mod_pth = analysis_cache_dir()
    with open(mod_pth / 'original_bias_model_fit.pkl', 'rb') as file:
        mod = pickle.load(file)


    ax = fig.add_subplot(gs[10:,0])

    # Apply FDR correction to the p-values of the fixed effects
    fdr_p = mod['pvals'].copy()
    fdr_p.iloc[:, :-2] = false_discovery_control(fdr_p.iloc[:, :-2], axis=0, method='bh')

    fdr_p = (fdr_p.rename(columns = {'Prior_Precision': rf'$\kappa_p$',
                                    'Coherence': rf'$\kappa_s$',
                                    'Coherence:Prior_Precision': rf'$\kappa_s$ x $\kappa_p$'})
                                    .drop(columns=['subjectNumber Var']))

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

    
    sns.despine()

    fig.text(0.01, 1, 'A', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.6, 'B', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.18, 'C', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    plt.tight_layout()

    plt.savefig(figure_output_path('fig07.svg'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    generate_fig07()
# %%
