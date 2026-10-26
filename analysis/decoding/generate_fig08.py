#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.regression.mixed_linear_model import MixedLM
from analysis.common import model_utils, tempgen_utils
from analysis.common.paths import figure_output_path
from scipy.stats import false_discovery_control
from analysis.common.plotting_helpers import pairwise_cluster_tests, plot_pvals, plot_timeseries_comparison, subject_level_summary
from pathlib import Path

TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20

min_dpi = 600
CM = 1/2.54

plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.3

plt.rcParams['lines.linewidth'] = 1.2

subplot_labelsize = 1

def generate_fig08():
    prior_colors = sns.color_palette('Oranges', n_colors=3)

    print('Plotting Temporal Generalisation split by Prior Precision')
    print('Loading temporal generalisation matrices... Please wait, this may take a moment as the data is quite large.')

    tempGenMats = tempgen_utils.load_tempgen_matrices_hdf() # shape = (n_subjects, n_prior_precisions, n_stim_coherences, n_timepoints, n_timepoints)

    time_in_seconds = (np.arange(577) / 256) - 0.25  # Convert timepoints to seconds

    # Average over stimulus coherence levels and training timepoints from 0.25 to 0.75 seconds post-stimulus onset.
    pk10 = tempGenMats[:, 2, :, ...].mean(axis = 1)[:,:,128:256].mean(axis=2) 
    pk5 = tempGenMats[:, 1, :, ...].mean(axis = 1)[:,:,128:256].mean(axis=2)
    pk0 = tempGenMats[:, 0, :, ...].mean(axis = 1)[:,:,128:256].mean(axis=2)

    # transform timeseries to dataframe for seaborn plotting
    arr = np.array([pk0, pk5, pk10])
    dim0, dim1, dim2 = np.indices(arr.shape)

    # Create long DataFrame
    long_df = pd.DataFrame({
        'Prior_Precision': dim0.flatten(),
        'subjectNumber': dim1.flatten(),
        'time': dim2.flatten(),
        'accuracy': arr.flatten()
    })

    # convert old indices to condition labels
    long_df['Prior_Precision'] = long_df['Prior_Precision'].map({0: 0, 1: 5, 2: 10})
    long_df['time'] = long_df['time']/256 - 0.25

    def decorate_timeseries(ax):
        ax.axvline(0.75, c='grey', linestyle='--')
        ax.axhline(0, c='grey', linestyle='--')
        ax.set(xlabel = 'Time (s)')

    # Now plot the data using the same workflow as before.
    fig  = plt.figure(figsize=(15,8))
    gs = GridSpec(7,2)

    ax = fig.add_subplot(gs[:5, 0])
    summary = subject_level_summary(long_df, cols = ['accuracy'], split_by = 'Prior_Precision')
    cluster_results = pairwise_cluster_tests(summary, split_by='Prior_Precision')

    # convert cluster_results to long dataframe for easier plotting
    cluster_results_df = {}
    for comparison, (clusters, p_vals) in cluster_results.items():
        cluster_results_df[comparison] = pd.DataFrame({
            'time': long_df['time'].unique(),
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

    plot_timeseries_comparison(summary = summary.query("time <= 0.75"),
                                cluster_results = cluster_results_obs, 
                            split_by = 'Prior_Precision',
                            smooth = True, 
                            plot_rt= False,
                            legend=False,
                            ax = ax,
                            palette=prior_colors)
    decorate_timeseries(ax)
    ax.axvline(0, c='red')

    ax.set_title('Observation Period', fontsize=PANEL_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('a.u.', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    ax = fig.add_subplot(gs[:5, 1])
    plot_timeseries_comparison(summary = summary.query("time > 0.75"),
                                cluster_results = cluster_results_resp,  
                            split_by = 'Prior_Precision',
                            smooth = True, 
                            plot_rt= False,
                            legend=True,
                            ax = ax,
                            palette=prior_colors)
    decorate_timeseries(ax)

    ax.set_title('Response Period', fontsize=PANEL_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

    # Run parallel mixedlm to get p-values for fixed effects for the plot of significant timepoints
    # Since this is subject-level data we run the model in real-time instead of loading the pre-fit trial-level models that take longer to fit. 
    params, pvals, resid = model_utils.run_parallel_mixedlm(df=long_df, dv='accuracy', n_jobs=10)

    # Apply FDR correction to the p-values of the fixed effects
    fdr_p = pvals.copy()
    fdr_p.iloc[:, :-2] = false_discovery_control(fdr_p.iloc[:, :-2], axis=0, method='bh')

    fdr_p = (fdr_p.rename(columns = {'Prior_Precision': rf'$\kappa_p$',
                                    })
                                    .drop(columns=['Group Var']))

    # Plot as before
    ax = fig.add_subplot(gs[5:,0])
    plot_pvals(fdr_p.query('time <= 0.75'),
            dv='Decoding Accuracy', 
            legend=False,
            intercept=False,
            color = prior_colors[1], 
            ax=ax)
    ax = fig.add_subplot(gs[5:,1])
    plot_pvals(fdr_p.query('time > 0.75'),
            dv='Decoding Accuracy', 
            legend=True, 
            intercept=False,
            color = prior_colors[1],
            ax=ax)

    sns.despine()
    plt.tight_layout()
    fig.text(0.01, 1, 'A', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.3, 'B', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')


    plt.savefig(figure_output_path('fig08.svg'),
                dpi=300,   
                bbox_inches='tight'
                )

if __name__ == "__main__":
    generate_fig08()
# %%
