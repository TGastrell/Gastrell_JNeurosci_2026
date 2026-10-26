import pandas as pd
from analysis.common import utils
import mne.stats
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot

def get_cmean_summary(df, cols):
    """
    Calculate summary statistics for a DataFrame grouped by specified columns.
    Included calcualtion normalised value column, enabling within subject standard error plotting.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cols (list): List of column names to group by. The last column in the list is treated as the value column. The first should be the ID.
    Returns:
        pd.DataFrame: A DataFrame containing the grouped mean and normalized mean values.
    Raises:
        ValueError: If cols is not a list or if it contains less than two elements.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a pandas DataFrame.")
    if not isinstance(cols, list):
        raise ValueError("cols should be a list of column names.")
    if len(cols) < 2:
        raise ValueError("cols should contain at least two column names for grouping.")

    summary = (df[cols]
           .groupby(cols[:-1])
           .agg(utils.cmean)
           .reset_index()
           .rename(columns = {f'{cols[-1]}':'cmean'})
           .dropna() # in some cases our exploratory factorisation leads to cells with no data (drawback of hamiltonian path counterbalancing approach)
           )
    
    summary = summary.merge(
        summary.groupby(cols[0])['cmean']
        .agg(utils.cmean)
        .reset_index()
        .rename(columns= {'cmean':'sub_avg_cmean'})
    )

    summary['grand_cmean'] = utils.cmean(summary.sub_avg_cmean)

    summary['cmean_norm'] = utils.circular_sum(
        utils.circular_difference(summary.cmean, summary.sub_avg_cmean), 
        summary.grand_cmean)
    return summary

def get_mean_summary(df, cols):
    """
    Calculate summary statistics for a DataFrame grouped by specified columns.
    Included calcualtion normalised value column, enabling within subject standard error plotting.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cols (list): List of column names to group by. The last column in the list is treated as the value column. The first should be the ID.
    Returns:
        pd.DataFrame: A DataFrame containing the grouped mean and normalized mean values.
    Raises:
        ValueError: If cols is not a list or if it contains less than two elements.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a pandas DataFrame.")
    if not isinstance(cols, list):
        raise ValueError("cols should be a list of column names.")
    if len(cols) < 2:
        raise ValueError("cols should contain at least two column names for grouping.")

    summary = (df[cols]
           .groupby(cols[:-1])
           .mean()
           .reset_index()
           .rename(columns = {f'{cols[-1]}':'cmean'})
           .dropna() # in some cases our exploratory factorisation leads to cells with no data (drawback of hamiltonian path counterbalancing approach)
           )
    
    summary = summary.merge(
        summary.groupby(cols[0])['cmean']
        .mean()
        .reset_index()
        .rename(columns= {'cmean':'sub_avg_cmean'})
    )

    summary['grand_cmean'] = np.mean(summary.sub_avg_cmean)

    summary['cmean_norm'] = (summary.cmean - summary.sub_avg_cmean) + summary.grand_cmean
    return summary

def get_cstd_summary(df, cols):
    """
    Calculate summary statistics for a DataFrame grouped by specified columns.
    Included calcualtion normalised value column, enabling within subject standard error plotting.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cols (list): List of column names to group by. The last column in the list is treated as the value column. The first should be the ID.
    Returns:
        pd.DataFrame: A DataFrame containing the grouped circular variance and normalized mean values.
    Raises:
        ValueError: If cols is not a list or if it contains less than two elements.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a pandas DataFrame.")
    if not isinstance(cols, list):
        raise ValueError("cols should be a list of column names.")
    if len(cols) < 2:
        raise ValueError("cols should contain at least two column names for grouping.")

    summary = (df[cols]
           .groupby(cols[:-1])
           .agg(utils.cstd)
           .reset_index()
           .rename(columns = {f'{cols[-1]}':'cstd'})
           )

    summary = summary.merge(
        summary.groupby(cols[0])['cstd']
        .agg(utils.cmean)
        .reset_index()
        .rename(columns= {'cstd':'sub_avg_cstd'})
    )
    grand_avg = summary.sub_avg_cstd.mean()

    summary['cstd_norm'] = summary.cstd - summary.sub_avg_cstd + grand_avg

    return summary

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
                        color = 'black', y_offset=0, alpha_thresh=0.05, label=''):
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
                   color=color, linewidth=3, alpha=0.8, label=c_label)
