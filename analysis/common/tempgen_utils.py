
import numpy as np
import pathlib
import scipy.stats as stats
from mne.stats import permutation_cluster_1samp_test
import h5py
from analysis.common.paths import analysis_cache_dir, derivatives_root


def _subject_directories(derivatives_pth):
    return sorted(
        (
            path
            for path in derivatives_pth.iterdir()
            if path.is_dir() and path.name.startswith('sub-')
        ),
        key=lambda path: path.name.lower(),
    )


def _tempgen_matrix_paths(derivatives_pth, subject_dirs, pk, sk, file_label):
    return [
        sub_dir
        / 'eeg'
        / 'FE'
        / 'train_sk2_unif'
        / 'tempGenMats'
        / f'trial-avgd-tempgen_matrix_trn-pk0_tst-pk{prior}sk{stim}_{file_label}.npy'
        for sub_dir in subject_dirs
        for prior in pk
        for stim in sk
    ]


def _aggregate_needs_refresh(aggregate_path, matrix_paths):
    if not aggregate_path.exists():
        return True

    aggregate_mtime = aggregate_path.stat().st_mtime_ns
    return any(
        matrix_path.exists() and matrix_path.stat().st_mtime_ns > aggregate_mtime
        for matrix_path in matrix_paths
    )


def load_tempgen_matrices(derivatives_pth=None):
    ''''
    Load the generated matrices for each subject, priorKappa, and stimKappa.

    Parameters
    ----------
    derivatives_pth : pathlib.Path
        Path to the derivatives directory containing the subject data.
    Returns
    -------
    np.ndarray
        Array of shape (n_subjects, n_priorKappa, n_stimKappa, n_time, n_time)
    '''
    if derivatives_pth is None:
        derivatives_pth = derivatives_root()

    pk = [0, 5, 10]
    sk = [0.2, 0.4, 2]
    n_pk = len(pk)
    n_sk = len(sk)
    file_label = 't--0.25-2_bl--0.35--0.25-0.2-40bpf-clean'
    
    n_time = 577  # number of time points
    all_mats = []

    pth = analysis_cache_dir() / 'allTempGenMats.npy'
    if not pth.exists():
        print('Aggregated temporal generalisation matrices not found, loading and processing individual subject data... \nThis only needs to be done once, subsequent runs will load the aggregated data and be much faster.')
        for sub_dir in _subject_directories(derivatives_pth):
            epoch_dir = sub_dir / 'eeg' / 'FE' / 'train_sk2_unif' / 'tempGenMats'   

            # Load the generated matrices for each subject, priorKappa, and stimKappa
            sub_mats= np.empty((n_pk, n_sk, n_time, n_time))  # Initialize with NaN for safety
            for p in range(n_pk):
                for s in range(n_sk):
                    gm = np.load(epoch_dir / f'trial-avgd-tempgen_matrix_trn-pk0_tst-pk{pk[p]}sk{sk[s]}_{file_label}.npy')
                    sub_mats[p, s, :, :] = gm # sub x pk x sk x time x time
            all_mats.append(sub_mats)

        all_mats = np.array(all_mats)  # Convert list to array of shape (n_subjects, n_priorKappa, n_stimKappa, n_time, n_time)
        pth.parent.mkdir(parents=True, exist_ok=True)
        np.save(pth, all_mats)
    else:
        print('Loading aggregated temporal generalisation matrices...')
        all_mats = np.load(pth)

    return np.array(all_mats)

def load_tempgen_matrices_hdf(derivatives_pth=None):
    '''
    Load the generated matrices for each subject, priorKappa, and stimKappa.
    Parameters
    ----------
    derivatives_pth : pathlib.Path
        Path to the derivatives directory containing the subject data.
    Returns
    -------
    np.ndarray
        Array of shape (n_subjects, n_priorKappa, n_stimKappa, n_time, n_time)
    '''
    if derivatives_pth is None:
        derivatives_pth = derivatives_root()

    pk = [0, 5, 10]
    sk = [0.2, 0.4, 2]
    n_pk = len(pk)
    n_sk = len(sk)
    file_label = 't--0.25-2_bl--0.35--0.25-0.2-40bpf-clean'

    n_time = 577
    pth = analysis_cache_dir() / 'allTempGenMats.h5'

    subject_dirs = _subject_directories(derivatives_pth)
    matrix_paths = _tempgen_matrix_paths(
        derivatives_pth, subject_dirs, pk, sk, file_label
    )
    if _aggregate_needs_refresh(pth, matrix_paths):
        if pth.exists():
            print('Subject temporal-generalisation matrices have changed; rebuilding the aggregate...')
        else:
            print('Aggregated temporal generalisation matrices not found, loading and processing individual subject data...'
                  '\nThis only needs to be done once, subsequent runs will load the aggregated data and be much faster.')

        all_mats = []
        for sub_dir in subject_dirs:
            epoch_dir = sub_dir / 'eeg' / 'FE' / 'train_sk2_unif' / 'tempGenMats'

            sub_mats = np.empty((n_pk, n_sk, n_time, n_time))
            for p in range(n_pk):
                for s in range(n_sk):
                    gm = np.load(epoch_dir / f'trial-avgd-tempgen_matrix_trn-pk0_tst-pk{pk[p]}sk{sk[s]}_{file_label}.npy')
                    sub_mats[p, s, :, :] = gm
            all_mats.append(sub_mats)

        all_mats = np.array(all_mats)

        pth.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(pth, 'w') as f:
            f.create_dataset(
                'all_mats',
                data=all_mats,
                compression='gzip',
                compression_opts=4,
                chunks=True  # auto-chunking, good default
            )
            # Store metadata as attributes — handy for later
            f['all_mats'].attrs['pk'] = pk
            f['all_mats'].attrs['sk'] = sk
    else:
        print('Loading aggregated temporal generalisation matrices...')

    with h5py.File(pth, 'r') as f:
        all_mats = f['all_mats'][:]

    return all_mats

def get_permutation_results(cond_mats, n_permutations=1000):
    '''
    Run a permutation test on the condition matrices to find significant clusters.
    Per Kok et al. 2017 and Moore et al. 2024, we use a p-value threshold of 0.01.

    See https://mne.tools/stable/generated/mne.stats.permutation_cluster_1samp_test.html
    for more details.
    
    Parameters
    ----------
    cond_mats : np.ndarray
        Condition matrices of shape (n_observations, n_time, n_time).
    n_permutations : int
        Number of permutations to run for the test.
    Returns
    -------
    T_obs : np.ndarray
        Observed t-statistics for each time point.
    clusters : list
        List of clusters found in the data.
    cluster_p_values : list
        p-values for each cluster.
    H0 : np.ndarray
        Null hypothesis distribution of the test statistic.
    '''
    # 
    n_observations = cond_mats.shape[0]  # number of subjects
    pval = 0.01  # per Kok et al. 2017 & Moore et al. 2024
    df = n_observations - 1  # degrees of freedom for the test
    thresh = stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    
    (T_obs, 
     clusters, 
     cluster_p_values, 
     H0) = permutation_cluster_1samp_test(
         cond_mats, 
         n_permutations=n_permutations,
         threshold=thresh, # threshold for cluster forming
         tail=0, # two-tailed test
         verbose=True, # Loading bar :)
         out_type = 'mask', # Get the cluster mask
         n_jobs=-1, # Use all available cores
         seed=42  # For reproducibility
         )
    
    return T_obs, clusters, cluster_p_values, H0
            
def get_significant_clusters(T_obs, clusters, cluster_p_values, alpha=0.05):
    '''
    Create a significance mask for the clusters found in the permutation test.
    This mask will be used to highlight significant clusters in the plots.
    Parameters
    ----------
    T_obs : np.ndarray  
        Observed t-statistics for each time point.
    clusters : list
        List of clusters found in the data.
    cluster_p_values : list
        p-values for each cluster.
    alpha : float
        Significance level for the test.
    Returns
    -------
    sig_mask : np.ndarray
        Boolean mask of significant clusters.
    '''
    # The mask will be of shape (n_time, n_time)
    height,width = T_obs.shape 
    
    sig_mask = np.zeros((height,width), dtype=bool)
        
    for i, (cluster, pval) in enumerate(zip(clusters, cluster_p_values)):
        if pval < alpha:
            sig_mask[cluster] = True

    return sig_mask
