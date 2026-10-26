import mne
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import false_discovery_control
from scipy import stats
from analysis.common.util_dataloader import load_decoding_data
from analysis.common import eeg_utils, tempgen_utils, utils
from statsmodels.regression.mixed_linear_model import MixedLM

TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20


def plot_single_timeseries(df, value, ax=None, color='black', is_circular=False, plot_clusters=False, plot_rt=False, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    if not is_circular:
        dat = (df
                .groupby(['subjectNumber', 'time'])[value]
                .mean()
                .reset_index()
                .pivot(index='subjectNumber', 
                        columns='time', 
                        values=value)
            )

        mu = gaussian_filter1d(np.mean(dat, axis=0), sigma=4)
        se = gaussian_filter1d(np.std(dat, axis=0)/np.sqrt(dat.shape[0]), sigma=4) # between SE for single timeseries
    else:
        dat = (df
                .groupby(['subjectNumber', 'time'])[value]
                .agg(lambda x: utils.cmean(x))
                .reset_index()
                .pivot(index='subjectNumber', 
                        columns='time', 
                        values=value)
            )
        mu = gaussian_filter1d(dat.apply(lambda x: utils.cmean(x), axis=0), sigma=4)
        se = gaussian_filter1d(np.std(dat, axis=0)/np.sqrt(dat.shape[0]), sigma=4) # within SE for circular data
    
    ax.plot(df.time.unique(),
            mu,
            color=color,
            **kwargs
            )
    ax.fill_between(x=df.time.unique(),
                    y1 = mu + se,
                    y2 = mu - se,
                    color=color,
                    alpha = 0.2)
    
    if plot_clusters:
        clusters, p_vals= run_cluster_test_1samp(dat, tail=0, n_permutations=10000)[1:3]
        plot_cluster_results(df.time.unique(), clusters, p_vals, ax,y_offset=-0.02,label='All v 0')

    if plot_rt:
        dat = (df
               .groupby(['subjectNumber'])['responseTime']
               .mean()
               .reset_index()
            )
            
        mu = dat['responseTime'].mean() + 0.85
        se = dat['responseTime'].std()/np.sqrt(len(dat))
        #ax.axvline(mu, color = palette[i], alpha = 0.3)
        ax.vlines(mu, ymin = 0, ymax=0.05, color = color, linestyles='--')
        #ax.scatter(mu, 0.15, color = color, marker = '|', s= 100)
        ax.plot([mu-se, mu+se],[0.025, 0.025], color = 'grey', linewidth = 1) # error bar
    
    sns.despine()
    
def subject_level_summary(_df, cols, split_by = 'Coherence', rt = False):
    if len(cols) > 1:
        _df['evk'] = _df[cols].mean(axis=1) # mean over electrode/column subset
    else:
        _df['evk'] = _df[cols]

    if not rt:
        _df['responseTime'] = 0

    summary = (_df
                .groupby(['subjectNumber', 'time', split_by])[['evk','responseTime']]
                .mean()
                .reset_index()
                )

    summary = summary.merge(
            _df.groupby(['subjectNumber', 'time'])[['evk','responseTime']]
            .mean()
            .reset_index()
            .rename(columns= {'evk':'sub_avg_evk',
                              'responseTime': 'sub_avg_rt' })
    )

    summary= summary.merge(
        summary.groupby(['time'])[['sub_avg_evk', 'sub_avg_rt']]
        .mean()
        .reset_index()
        .rename(columns={'sub_avg_evk':'grand_mean',
                         'sub_avg_rt' : 'grand_rt'})
    )

    summary['evk_norm'] = summary['evk'] - summary['sub_avg_evk'] + summary['grand_mean']
    summary['rt_norm'] = summary['responseTime'] - summary['sub_avg_rt'] + summary['grand_rt']
    return summary

def subject_level_summary_circular(_df, cols, split_by = 'Coherence', rt = False):
    if len(cols) > 1:
        _df['evk'] = _df[cols].mean(axis=1) # mean over electrode/column subset
    else:
        _df['evk'] = _df[cols]

    if not rt:
        _df['responseTime'] = 0

    summary = (_df
                .groupby(['subjectNumber', 'time', split_by])[['evk']]
                .agg(lambda x: utils.cmean(x))
                .reset_index()
                )

    summary = summary.merge(
            _df.groupby(['subjectNumber', 'time'])[['evk']]
            .agg(lambda x: utils.cmean(x))
            .reset_index()
            .rename(columns= {'evk':'sub_avg_evk'})
    )

    summary= summary.merge(
        summary.groupby(['time'])[['sub_avg_evk']]
        .agg(lambda x: utils.cmean(x))
        .reset_index()
        .rename(columns={'sub_avg_evk':'grand_mean'})
    )

    summary['evk_norm'] = utils.circular_sum(utils.circular_difference(summary['evk'], summary['sub_avg_evk']), summary['grand_mean'])
    return summary

def plot_timeseries_comparison(summary,cluster_results=None,split_by = 'Coherence', plot_rt = False, smooth = False, is_circular = False, legend = False, ylim = None, ax = None, palette = None):
    ''' 
    Plot timeseries comparison with cluster-based significance testing.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'subjectNumber', 'time', cols, split_by
    cols : list
        List of columns to average over (e.g., electrodes)
    split_by : str
        Column name to split the data by (e.g., 'Coherence' or 'Prior_Precision')
    ax : matplotlib.axes.Axes
        Axes to plot on
    Returns
    -------
    None
    '''
    if ax is None:
        ax = plt.gca()
    if palette is None:
        palette = sns.color_palette("Blues", n_colors=3)

    cluster_styles = ['solid', 'dashed', 'dotted']

    for i, level in enumerate(summary[split_by].unique()):
        query = f'{split_by} == {level}'
        level_dat = summary.query(query)
        
        if not is_circular:
            dat = (level_dat
            .groupby(['subjectNumber', 'time'])['evk_norm']
            .mean()
            .reset_index()
            .pivot(index='subjectNumber', 
                    columns='time', 
                    values='evk_norm')
                )
            mu = gaussian_filter1d(np.mean(dat, axis= 0), 4, 0)
            se = gaussian_filter1d(np.std(dat, axis = 0) / np.sqrt(dat.shape[0]), 4, 0) # within SE
        else:
            dat = (level_dat
            .groupby(['subjectNumber', 'time'])['evk_norm']
            .agg(lambda x: utils.cmean(x))
            .reset_index()
            .pivot(index='subjectNumber', 
                    columns='time', 
                    values='evk_norm')
                )
            mu = gaussian_filter1d(dat.apply(lambda x: utils.cmean(x), axis=0), 4, 0)
            se = gaussian_filter1d(np.std(dat, axis=0)/np.sqrt(dat.shape[0]), 4, 0) # within SE for circular data

        ax.plot(level_dat.time.unique(),
                mu,
                color = palette[i],
                label = level)
        
        ax.fill_between(level_dat.time.unique(),
                        mu + se,
                        mu - se,
                        color = palette[i],
                        alpha = 0.2)
        
        if not plot_rt:
            continue
        
        dat = (level_dat
        .groupby(['subjectNumber'])['rt_norm']
        .mean()
        .reset_index()
            )

        mu = dat['rt_norm'].mean() + 0.85
        se = dat['rt_norm'].std()/np.sqrt(len(dat))

        #ax.axvline(mu, color = palette[i], alpha = 0.3)
        #ax.scatter(mu, 0.0, color = palette[i], marker = '|', s= 100)
        ax.vlines(mu, ymin = 0, ymax=0.05, color = palette[i], linestyles='--')
        ax.plot([mu-se, mu+se],[0.025, 0.025], color = 'grey', linewidth = 1) # error bar

    if cluster_results is not None:
        for i, (comparison, c) in enumerate(cluster_results.items()):
            y_offset= -0.02-i*0.005 if  not is_circular else -0.4 - i*0.05
            sig = c.query('p_val < 0.05')['time']
            if len(sig) == 0:
                continue
            ax.plot(sig, 
                    (y_offset) * np.ones(len(sig)), 
                    color='black', linestyle = cluster_styles[i], linewidth=2, alpha=0.8, 
                    label=comparison
                    )

    if ylim is not None:
        ax.set_ylim(ylim)
    if legend:
        title = r'Stimulus Precision ($\kappa_s$)' if split_by == 'Coherence' else r'Prior Precision ($\kappa_p$)'
        ax.legend(bbox_to_anchor = (1.01, 1), title=title, frameon=False, fontsize=PANEL_LABEL_SIZE, title_fontsize=PANEL_LABEL_SIZE)
    
    #sns.despine(trim=True)

def plot_tempGen(all_mats, ax=None, n_permutations=1000, rt = 1.85):
    if ax is None:
        ax = plt.gca()

    alpha = 0.05  # significance threshold
    n_time = all_mats.shape[-1]
    time_in_seconds = (np.arange(n_time) / 256) - 0.25  # Convert timepoints to seconds
    T_obs, clusters, cluster_p_values, H0 = tempgen_utils.get_permutation_results(all_mats, n_permutations=n_permutations)
    sig_mask = tempgen_utils.get_significant_clusters(T_obs, clusters, cluster_p_values, alpha)

    all_mats_grand = np.mean(all_mats, axis=0)
    sampling_frequency = 256  # in Hz

    # Apply Gaussian filter to smooth the data
    sm_mat = gaussian_filter(all_mats_grand, 2) 

    im = ax.imshow(sm_mat.T,  
                    cmap='viridis', 

                    extent = [time_in_seconds[0], 
                                time_in_seconds[-1], 
                                time_in_seconds[0], 
                                time_in_seconds[-1]],  # Set extent to match time in seconds
                    aspect='auto',
                    origin='lower',
                    vmin = -0.08,
                    vmax = 0.12
                    )
    # add colorbar
    cbar = plt.colorbar(im, ax=ax, label='t-value')
    cbar.set_label('t-value', fontsize=16)  # Set label with fontsize
    cbar.ax.tick_params(labelsize=14)  # Set tick label fontsize
    # Create coordinate meshgrids for contour
    X, Y = np.meshgrid(time_in_seconds, time_in_seconds)
    ax.contour(X,Y,sig_mask.T,levels=[0.5], colors='white')
    ax.axvline(x=0, color = 'red')
    ax.axhline(y=0, color = 'red')
    ax.axvline(x=0.75, color = 'black', linestyle='--', alpha=0.5)
    ax.axhline(y=0.75, color = 'black', linestyle='--', alpha=0.5)
    #ax.axvline(x=rt, color = 'grey', alpha=0.5)
    #ax.axhline(y=rt, color = 'grey', alpha=0.5)
    ax.set_xlabel('Test Time', fontsize=16)
    ax.set_ylabel("Train Time", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)


def pairwise_cluster_tests(summary, split_by):
    '''
    Run pairwise cluster-based permutation tests for differences between levels of a factor.
    Uses a fast implementation that does not require pre-computing an adjacency matrix, but is limited to 1D data (e.g., time series).
    
    Parameters
    ----------
    summary : pd.DataFrame
        The summary data frame
    split_by : str
        The column name to split by

    Returns
    -------
    cluster_results : dict
        A dictionary containing the cluster test results
    '''
    # Get the data in the right format
    for_test = {}
    levels = summary[split_by].unique()
    for level in levels:
        for_test[level] = np.array(summary
                           .query(f'{split_by} == @level')[['subjectNumber','time', 'evk']]
                           .pivot(index = 'subjectNumber', 
                                  columns = 'time', 
                                  values = 'evk')
        )

    # Run the cluster tests (fast)
    cluster_results = {}
    cluster_results[f'{levels[2]:.1f} v {levels[0]:.1f}'] = run_cluster_test_1samp(np.array(for_test[levels[2]] - for_test[levels[0]]),
                                                                            tail=0, n_permutations=10000)[1:3]
    cluster_results[f'{levels[2]:.1f} v {levels[1]:.1f}'] = run_cluster_test_1samp(np.array(for_test[levels[2]] - for_test[levels[1]]),
                                                                            tail=0, n_permutations=10000)[1:3]
    cluster_results[f'{levels[1]:.1f} v {levels[0]:.1f}'] = run_cluster_test_1samp(np.array(for_test[levels[1]] - for_test[levels[0]]),
                                                                                tail=0, n_permutations=10000)[1:3]
    return cluster_results

def run_cluster_test_1samp(data, tail=0, n_permutations=10000, adjacency=None):
    """
    Run 1-sample permutation cluster test against zero.
    
    Parameters:
    -----------
    data : array, shape (n_subjects, n_times)
        Data to test
    tail : int
        -1 for negative, 0 for two-tailed, 1 for positive
    alpha : float
        Significance threshold
    n_permutations : int
        Number of permutations
    adjacency : array or None
        Adjacency matrix for clustering (None for temporal adjacency)
    
    Returns:
    --------
    T_obs : array
        T-statistics
    clusters : list
        List of clusters
    cluster_p_values : array
        P-values for each cluster
    H0 : array
        Null distribution
    """
    n_observations = data.shape[0]  # number of subjects
    pval = 0.01  # .01 per Kok et al. 2017 & Moore et al. 2024
    df = n_observations - 1  # degrees of freedom for the test
    
    if tail not in [-1, 0, 1]:
        raise ValueError("tail must be -1, 0, or 1")

    match tail:
        case 0:
            thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
        case 1:
            thresh = stats.t.ppf(1 - pval, df)  # one-tailed, t distribution
        case -1:
            thresh = stats.t.ppf(pval, df)  # one-tailed, t distribution
    
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        data, 
        n_permutations=n_permutations,
        tail=tail, 
        threshold=thresh,  # threshold for cluster forming
        n_jobs=-1,  # Use all available cores
        seed=42,  # For reproducibility
        adjacency=adjacency,
        verbose=False
    )
    
    return T_obs, clusters, cluster_p_values, H0

def plot_cluster_results(time_in_seconds,clusters, cluster_p_values, ax,
                        color = 'black', style = 'solid', y_offset=0, alpha_thresh=0.05, label=''):
    """
    Plot results with cluster-based significance testing.
    """ 
    # Initialize label_used to track if we've used the label
    label_used = False

    for i_c, c in enumerate(clusters):
        c = c[0]  # Get the cluster indices
        if cluster_p_values[i_c] <= alpha_thresh:
            # Only use label for the first significant cluster
            if not label_used:
                c_label = label
                label_used = True
            else:
                c_label = '_nolegend_'  # Hide subsequent entries from legend
                
            ax.plot(time_in_seconds[c], 
                   (y_offset) * np.ones(len(time_in_seconds[c])), 
                   color=color, linestyle = style, linewidth=2, alpha=0.8, label=c_label)

def plot_pvals(fdr_p, dv='CPP', legend = True, intercept = False,ax=None, color=None):
    ''' Plot FDR-corrected p-values 
    
    Parameters
    ----------
    fdr_p : pd.DataFrame
        DataFrame with FDR-corrected p-values for each time point and predictor.
    '''
    if ax is None:
        ax = plt.gca()

    for col in fdr_p.columns:
        if col == 'Intercept' and intercept == False:
            continue
        elif col == 'time':
            continue
        ax.plot(fdr_p.time, gaussian_filter1d(-np.log(fdr_p[col], ), 4, 0), label=col, 
                color=color
                )
    
    ax.axhline(-np.log(0.05), color='grey', linestyle='--', label = '$p_{FDR}$ < .05')

    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.set_ylabel(r'-$\log_(p_{FDR})$', fontsize=AX_LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)


    #sns.despine(trim=True)
    if legend:
        plt.legend(title = 'Predictor', bbox_to_anchor=(1.01, 1.2), frameon=False, fontsize=PANEL_LABEL_SIZE, title_fontsize=PANEL_LABEL_SIZE)