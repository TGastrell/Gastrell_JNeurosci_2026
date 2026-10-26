#%%
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
from joblib import Parallel, delayed
from pathlib import Path
import pickle
from analysis.common import eeg_utils, utils
from analysis.common.paths import analysis_cache_dir, derivatives_root, figure_output_path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import false_discovery_control
from scipy import stats
from matplotlib.gridspec import GridSpec

stim_colors = sns.color_palette("Blues", n_colors=3)
prior_colors = sns.color_palette("Oranges", n_colors=3)
TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20

def df_to_evk(df, info, tmin):
    ''' Convert DataFrame back to MNE Evoked object '''
    evk = mne.EvokedArray(
        data = df.T.values,
        info = info,
        tmin = tmin
    )
    return evk

def pairwise_cluster_tests(summary, split_by='stimKappa'):
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

def plot_cluster_results(time_in_seconds, clusters, cluster_p_values, ax,
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
            ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)

def plot_timeseries_comparison(_df, cols, split_by = 'Coherence', plot_rt = False, smooth = False, legend = False, ylim = None, ax = None):
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
    palette = stim_colors if split_by == 'Coherence' else prior_colors
    cluster_styles = ['solid', 'dashed', 'dotted']

    if len(cols) > 1:
        _df['evk'] = _df[cols].mean(axis=1) # mean over electrode/column subset
    else:
        _df['evk'] = _df[cols]

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
    

    for i, level in enumerate(summary[split_by].unique()):
        query = f'{split_by} == {level}'
        level_dat = summary.query(query)
        
        dat = (level_dat
        .groupby(['subjectNumber', 'time'])['evk_norm']
        .mean()
        .reset_index()
        .pivot(index='subjectNumber', 
                columns='time', 
                values='evk_norm')
            )
        mu = gaussian_filter1d(np.mean(dat, axis= 0), 4, 0) if smooth else np.mean(dat, axis= 0)
        se = gaussian_filter1d(np.std(dat, axis = 0) / np.sqrt(dat.shape[0]), 4, 0) if smooth else np.std(dat, axis = 0) / np.sqrt(dat.shape[0])# within SE

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
        #ax.scatter(mu, 0.15, color = palette[i])
        ax.plot([mu-se, mu+se],[0.15, 0.15], color = palette[i], linewidth = 2) # error bar

    # Run cluster comparison tests
    cluster_results = pairwise_cluster_tests(summary, split_by=split_by)

    # Plot the significant clusters
    for i, (comparison, (clusters, cluster_p_values)) in enumerate(cluster_results.items()):
        plot_cluster_results(summary.time.unique(), clusters, cluster_p_values, ax,
                            style = cluster_styles[i], y_offset= -0.1-i*0.05 , alpha_thresh=0.05, label= comparison)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    if legend:
        title = r'Stimulus Precision ($\kappa_s$)' if split_by == 'Coherence' else r'Prior Precision ($\kappa_p$)'
        ax.legend(bbox_to_anchor = (1, 1.), title=title, frameon=False, fontsize=AX_LABEL_SIZE, title_fontsize=AX_LABEL_SIZE)
    
    #sns.despine(trim=True)

def plot_ERP(df, electrode_subset, split_by = 'Coherence', ax = None):
    palette = stim_colors if split_by == 'Coherence' else prior_colors
    cluster_styles = ['solid', 'dashed', 'dotted']

    _df = df.copy()
    if len(electrode_subset) > 1:
        _df['evk'] = _df[electrode_subset].mean(axis=1) # mean over electrode subset
    else:
        _df['evk'] = _df[electrode_subset]

    summary = (_df
                .groupby(['subjectNumber', 'time', split_by])['evk']
                .mean()
                .reset_index()
                )

    summary = summary.merge(
            _df.groupby(['subjectNumber', 'time'])['evk']
            .mean()
            .reset_index()
            .rename(columns= {'evk':'sub_avg_evk'})
    )

    summary= summary.merge(
        summary.groupby(['time'])['sub_avg_evk']
        .mean()
        .reset_index()
        .rename(columns={'sub_avg_evk':'grand_mean'})
    )

    summary['evk_norm'] = summary['evk'] - summary['sub_avg_evk'] + summary['grand_mean']

    # Run cluster tests
    cluster_results = pairwise_cluster_tests(summary, split_by=split_by)
    
    # Plot ERP 
    sns.lineplot(
        data = summary,
        x = 'time',
        y = 'evk_norm',
        hue = split_by,
        palette = palette,
        ax = ax
    )

    # Plot the significant clusters
    for i, (comparison, (clusters, cluster_p_values)) in enumerate(cluster_results.items()):
        plot_cluster_results(summary.time.unique(), clusters, cluster_p_values, ax,
                            style = cluster_styles[i], y_offset= - 0.3- i*0.05 , alpha_thresh=0.05, label= comparison)
    
    # Customize plot
    ax = plt.gca()
    ax.axvline(0, c='red')
   # ax.axvline(0.75, c='grey', linestyle='--')
    ax.set(ylabel=r'$\mu$V',
           xlabel='Time (s)',
           ylim = (-0.5, 1.2))
    sns.despine(trim=True)

    title = r'Coherence ($\kappa_s$)' if split_by == 'Coherence' else r'Prior Precision ($\kappa_p$)'
    ax.legend(bbox_to_anchor = (1, 1.01), loc='upper right', title=title, frameon=False, fontsize=AX_LABEL_SIZE, title_fontsize=AX_LABEL_SIZE)


def plot_pvals(all_pvals, dv='CPP', legend = True, intercept = False,ax=None, topPanel = False):
    ''' Plot FDR-corrected p-values 
    
    Parameters
    ----------
    all_pvals : pd.DataFrame
        DataFrame with p-values for each time point and predictor.
    '''
    if ax is None:
        ax = plt.gca()

    fdr_p = all_pvals.iloc[:, :-2].copy()
    fdr_p.iloc[:] = false_discovery_control(fdr_p, axis=0, method='bh')
    
    fdr_p = fdr_p.rename(columns = {'Prior_Precision': rf'$\kappa_p$',
                                    'Coherence': rf'$\kappa_s$',
                                    'Coherence:Prior_Precision': rf'$\kappa_s$ x $\kappa_p$'})
    for col in fdr_p.columns:
        if col == 'Intercept' and intercept == False:
            continue
        if topPanel:
            if col == 'Coherence':
                alpha = 1
            else:
                alpha = 0
        else:
            if col == 'Coherence':
                alpha = 0
            else:
                alpha = 1

    
        ax.plot(all_pvals.time, gaussian_filter1d(-np.log(fdr_p[col]), 4, 0), label=col, alpha=alpha)
    
    ax.axhline(-np.log(0.05), color='grey', linestyle='--', label = '$p_{FDR}$ < .05')

    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.set_ylabel(r'-$\log_(p_{FDR})$', fontsize=AX_LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    
    if legend:
        plt.legend(title = 'Predictor', bbox_to_anchor=(1., 1.2), frameon=False, fontsize=AX_LABEL_SIZE, title_fontsize=AX_LABEL_SIZE)

        
def fit_mixedlm(df, formula, groups):
    model = MixedLM.from_formula(formula, groups=groups, data=df)
    result = model.fit(method='nm', maxiter=200)
    return result

def fit_timepoint(t, df_t, dv='cpp'):
    print(f'Fitting time {t}')
    formula = f'{dv} ~ Coherence * Prior_Precision'
    groups = df_t['subjectNumber']
    result = fit_mixedlm(df_t, formula, groups)
    return pd.DataFrame(result.params).T.assign(time=t), pd.DataFrame(result.pvalues).T.assign(time=t), pd.DataFrame(result.resid).assign(time=t)

def run_parallel_mixedlm(df, dv, n_jobs=10):
    results = Parallel(n_jobs=n_jobs, 
                         verbose=10,
                         batch_size=5)(delayed(fit_timepoint)(t, df.query('time == @t')[[dv, 'Coherence', 'Prior_Precision', 'subjectNumber']], dv=dv) for t in df.time.unique())
    params = {}
    pvals = {}
    residuals = {}
    for i, t_results in enumerate(results):
        params[i] = t_results[0]
        pvals[i] = t_results[1]
        residuals[i] = t_results[2]

    params = pd.concat(params, ignore_index=True)
    pvals = pd.concat(pvals, ignore_index=True)
    resid = pd.concat(residuals, ignore_index=False)
    return params, pvals, resid

def decorate_timeseries(ax):
    ax.axvline(0, c='r')
    ax.axvline(0.75, c='grey', linestyle='--')
    ax.set_xlabel('Time (s)', fontsize=AX_LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
#%%
if __name__ == '__main__':
    derivatives_pth = derivatives_root()

    file_label='t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean'

    filename = analysis_cache_dir() / f'trial_avg_cpp_long_{file_label}_for_mixedlm.parquet'

    if not filename.exists():
        t_min = -0.95
        t_max = 2
        bl_min = -0.35
        bl_max = -0.25
        bpf_low = 0.2
        bpf_high = 40

        sorted_pts = sorted(derivatives_pth.iterdir(), key=lambda x: x.name.lower())
        df, electrodes = eeg_utils.load_all_epochs_with_metadata(sorted_pts, t_min, t_max, 
                                                bl_min, bl_max, bpf_low, bpf_high, eogs=True,
                                                electrode_subset=['CP1', 'CPz', 'CP2', 'Cz', 'Pz'],
                                                crop=(0, 0.75))
        

        # convert all float64 to float32 to save memory
        float64_cols = df.select_dtypes(include=['float64']).columns
        df[float64_cols] = df[float64_cols].astype('float32')
        df['responseError'] = utils.circular_difference(df['responseAngle'], df['stimDir'])
        df['prior_stim_distance'] = utils.circular_difference(df['priorDir'], df['stimDir'])
        df['signed_error'] = utils.get_signed_error(df['responseError'], df['prior_stim_distance'])

        df['cpp'] = df[electrodes].mean(axis=1)

        df = (df[['subjectNumber', 'priorKappa', 'stimKappa', 'time', 'cpp', 'responseTime']]
                .groupby(['subjectNumber',
                        'time', 
                        'stimKappa',
                        'priorKappa'])
                .mean()
                .reset_index())

        df = df.rename(columns={'stimKappa': 'Coherence', 
                            'priorKappa': 'Prior_Precision'})
        

        df.to_parquet(filename)
    else:
        print('Loading preprocessed EEG data... ')
        df = pd.read_parquet(filename)
        df = df.rename(columns={'time_secs': 'time'})

    dv = 'cpp' 

    fit_dir = analysis_cache_dir() / 'cpp_model_fit.pkl'
    if fit_dir.exists():
        print('Loading modelling results... ')
        with open(fit_dir, 'rb') as f:
            fit_dict = pickle.load(f)
        params = fit_dict['params']
        pvals = fit_dict['pvals']
        resid = fit_dict['resid']
    else:
        print('Modelling results not found. Go run it and come back :)')

    print('Done! Now plotting results... ')
    
    fig = plt.figure(figsize=(15,14))
    gs = GridSpec(12,1)

    print('Plotting timeseries comparisons for Stimulus Precision... ')
    ax1 = fig.add_subplot(gs[:5,0])
    plot_timeseries_comparison(df, ['cpp'], split_by='Coherence', ax=ax1, legend=True, smooth = True)
    decorate_timeseries(ax1)
    ax1.set(ylim = (-0.3, 1.1))
    ax1.set_ylabel(r'$\mu$V', fontsize=AX_LABEL_SIZE)
    ax1.set_title('CPP Amplitudes', fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    
    print('Plotting timeseries comparisons for Prior Precision... ')
    ax2 = fig.add_subplot(gs[5:10,0])
    plot_timeseries_comparison(df, ['cpp'], split_by='Prior_Precision', ax=ax2, legend = True, smooth = True)
    decorate_timeseries(ax2)
    ax2.set(ylim = (-0.3, 1.1))
    ax2.set_ylabel(r'$\mu$V', fontsize=AX_LABEL_SIZE)
    #ax2.set_title('CPP Amplitudes', fontsize=PANEL_LABEL_SIZE, fontweight='bold')

    print('Plotting p-values for mixed effects model... ')
    ax4 = fig.add_subplot(gs[10:,0])
    plot_pvals(pvals, 'CPP', ax=ax4, legend = True, topPanel = False)
    decorate_timeseries(ax4)
    ax4.set_yscale('log')
    ax4.set(ylim = (0, 10e2))

    print('Adding panel labels and saving figure... ')
    fig.text(0.02, 0.98, 'A', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.02, 0.6, 'B', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.02, 0.2, 'C', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    sns.despine()
    plt.tight_layout()

    plt.savefig(figure_output_path('fig04.svg'), dpi=300)
# %%
