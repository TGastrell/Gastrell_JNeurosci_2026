'''
This script estimates the sensitivity of each electrode to motion information using a Mixed Linear Model (MixedLM).
The motion information is decomposed into sine and cosine components, and the beta coefficients are estimated for each component. The vector length of these coefficients is used as an index of sensitivity.
The script also performs a bootstrap procedure to create an empirical null distribution of sensitivity values by randomizing the motion information and re-estimating sensitivity multiple times. This allows for statistical inference on the observed sensitivity values.
The script is designed to run in parallel across electrodes and timepoints for efficiency.
Usage:
    python MixedLM_sensors.py <path_to_derivatives_folder> <path_to_output>
'''

from analysis.common.eeg_utils import load_all_epochs
from analysis.common.paths import analysis_cache_dir, derivatives_root
import mne
import sys
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from joblib import Parallel, delayed
from pathlib import Path
from scipy.stats import norm   
import pickle 
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy import linalg

def estimate_electrode_motion_sens(dv, design, id):
    '''
    Estimate the sensitivity of an electrode (dv) to motion information decomposed into sin and cos.
    Fits a MixedLM and estimates the beta coefficients for each component. 
    Returns the vector length as an index of sensitivity
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        lm = MixedLM(dv, design, groups = id)
        result = lm.fit(method='nm')

        sin_beta, cos_beta = result.params['sin_theta'], result.params['cos_theta']

    return np.sqrt(sin_beta**2 + cos_beta**2)

def fast_within_subject_regression(dv, design, subject_ids):
    """
    Fast alternative to MixedLM using within-subject demeaning.
    This removes random intercepts without iterative optimization.
    
    Mathematically equivalent to fixed-effects model.
    """

    design_sin = design['sin_theta']
    design_cos = design['cos_theta']

    # Get unique subjects
    unique_subjects = np.unique(subject_ids)
    
    # Pre-allocate demeaned arrays
    dv_dm = np.zeros_like(dv, dtype=np.float32)
    sin_dm = np.zeros_like(design_sin, dtype=np.float32)
    cos_dm = np.zeros_like(design_cos, dtype=np.float32)
    
    # Demean within each subject (removes random intercept)
    for subj in unique_subjects:
        mask = subject_ids == subj
        dv_dm[mask] = dv[mask] - np.mean(dv[mask])
        sin_dm[mask] = design_sin[mask] - np.mean(design_sin[mask])
        cos_dm[mask] = design_cos[mask] - np.mean(design_cos[mask])
    
    # Stack predictors
    X = np.column_stack([sin_dm, cos_dm])
    
    # Fast least squares using Cholesky decomposition
    XtX = X.T @ X
    Xty = X.T @ dv_dm
    
    try:
        # Cholesky is fastest for positive definite matrices
        L = linalg.cholesky(XtX, lower=True, check_finite=False)
        beta = linalg.solve_triangular(L, Xty, lower=True, check_finite=False)
        beta = linalg.solve_triangular(L.T, beta, lower=False, check_finite=False)
    except linalg.LinAlgError:
        # Fallback if matrix is singular
        beta = linalg.lstsq(X, dv_dm, check_finite=False)[0]
    
    sin_beta, cos_beta = beta[0], beta[1]
    return np.sqrt(sin_beta**2 + cos_beta**2)

def bootstrap_empirical_null(dv, design, id, n_permutations = 1000):
    empirical_null = []
    for p in range(n_permutations):
        design_shuffled = design.sample(frac=1, random_state = p).reset_index(drop=True) # randomise the motion information

        permuted_sens = estimate_electrode_motion_sens(dv, design_shuffled, id) # recompute sensitivity matrix
        empirical_null.append(permuted_sens)
    
    return np.stack(empirical_null)

def bootstrap_empirical_null_adaptive(dv, design, id, obs_stat ,min_perm = 100, max_perm = 1000):
    
    initial_null = np.zeros(min_perm, dtype=np.float32)

    for p in range(min_perm):
        design_shuffled = design.sample(frac=1, random_state = p).reset_index(drop=True) # randomise the motion information

        initial_null[p] = fast_within_subject_regression(dv, design_shuffled, id) # recompute sensitivity using faster method

    quick_p = np.mean(initial_null >= obs_stat) # what proportion of the null is >= to the obs?

    # if it's obviously a lot, then we don't need to keep permuting. 
    if quick_p > 0.1:
        rng_fill = np.random.default_rng(42)
        return rng_fill.choice(initial_null, size=max_perm, replace=True) # resample with replacement up to the full max_perm
    
    remaining_perm = max_perm-min_perm
    additional_null = np.zeros(remaining_perm, dtype=np.float32)
    for p in range(remaining_perm):
        design_shuffled = design.sample(frac=1, random_state = p).reset_index(drop=True)

        additional_null[p] = fast_within_subject_regression(dv, design_shuffled, id) 
    return np.concatenate([initial_null, additional_null])

def parallel_job(dv, design, id):
    # fitting
    print('fitting...')
    sens = estimate_electrode_motion_sens(dv, design, id)
    print('bootstrapping...')
    empirical_null = bootstrap_empirical_null_adaptive(dv, design, id, min_perm=100, max_perm=1000, obs_stat=sens)

    return sens, empirical_null

def run_parallel_estiamtion(data: pd.DataFrame, electrode: str, design: pd.DataFrame, id, timepoints, n_jobs = -1):
    results = Parallel(n_jobs = n_jobs, backend='loky', verbose=10)(
        delayed(parallel_job)
        (data.query('time == @t')[electrode].reset_index(drop = True), 
         design,
         id)
        for t in timepoints
        )
    return zip(*results)

def estimate_stats(obs, null, two_sided):
    '''
    obs : electrodes x timepoints
    null : electrodes x timepoints x null_samples
    '''
    n_null_samples = null.shape[-1]
    
    # empirical percentile, add 0.5/N continuity correction
    k = np.sum(null <= obs[..., None], axis=-1) 
    
    percentiles = (k + 0.5) / (n_null_samples + 1)

    if two_sided:
        # two-tailed p-value
        p_values = 2 * np.min(percentiles, 1 - percentiles)
        z_scores = norm.isf(p_values / 2)  # inverse survival function
    else:
        # one-tailed
        p_values = 1 - percentiles
        z_scores = norm.ppf(percentiles)

    return percentiles, p_values, z_scores

def process_single_electrode_timepoint(data_slice, design, id):
    """Single job: one electrode, one timepoint"""
    sens = estimate_electrode_motion_sens(data_slice, design, id)
    empirical_null = bootstrap_empirical_null_adaptive(data_slice, design, id, min_perm=100, max_perm=1000, obs_stat=sens)
    return sens, empirical_null

def run_parallel_estimation_optimized(data: pd.DataFrame, electrodes, design: pd.DataFrame, 
                                     id, timepoints, n_jobs=-1, backend='loky', 
                                     batch_size='auto', verbose=10):
    """
    Parallelizes over BOTH electrodes and timepoints for maximum efficiency.
    
    Parameters:
    -----------
    backend : str
        'loky' (default, best for HPC), 'multiprocessing', or 'threading'
    batch_size : int or 'auto'
        Number of tasks per batch. 'auto' lets joblib decide
    verbose : int
        Verbosity level for joblib progress tracking
    """
    
    # Create all electrode-timepoint combinations
    jobs = [(e, t) for e in electrodes for t in timepoints]
    n_jobs_total = len(jobs)
    
    print(f"Running {n_jobs_total} parallel jobs ({len(electrodes)} electrodes × {len(timepoints)} timepoints)")
    print(f"Using backend: {backend}, n_jobs: {n_jobs}")
    
    # Parallel execution over all electrode-timepoint pairs
    results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size, verbose=verbose)(
        delayed(process_single_electrode_timepoint)(
            data.query('time == @t')[e].reset_index(drop=True),
            design,
            id
        )
        for e, t in jobs
    )
    
    # Reshape results into matrices
    sens_matrix = np.zeros((len(electrodes), len(timepoints)))
    empirical_null_matrix = np.zeros((len(electrodes), len(timepoints), 1000))
    
    for idx, (e_idx, t_idx) in enumerate([(electrodes.index(e), list(timepoints).index(t)) 
                                            for e, t in jobs]):
        sens, emp_null = results[idx]
        sens_matrix[e_idx, t_idx] = sens
        empirical_null_matrix[e_idx, t_idx, :] = emp_null
    
    return sens_matrix, empirical_null_matrix

if __name__ == "__main__":
    
    derivatives_pth = derivatives_root()
    out_pth = analysis_cache_dir()

    t_min = -0.95
    t_max = 2
    bl_min = -0.35
    bl_max = -0.25
    bpf_low = 0.2
    bpf_high = 40
    
    # HPC configuration
    N_JOBS = -1  # Use all available cores, or set to specific number (e.g., 32, 64)
    BACKEND = 'loky'  # Best for HPC, use 'multiprocessing' if loky has issues
    
    sorted_pts = sorted(derivatives_pth.iterdir(), key=lambda x: x.name.lower())
    df, metadata, electrodes = load_all_epochs(sorted_pts, t_min, t_max, 
                                               bl_min, bl_max, bpf_low, bpf_high, eogs=True,
                                               filter_query = 'stimKappa == 2.0 & priorKappa == 0')
    
    df[electrodes] = df[electrodes].astype('float32') # halve memory usage :)

    timepoints = df['time'].unique()
    
    predictors = ['sin_theta', 'cos_theta']
    design = metadata.copy()
    design['sin_theta'] = np.sin(design['stimDir'])
    design['cos_theta'] = np.cos(design['stimDir'])
    design = design[predictors].reset_index(drop=True)
    
    print(f'Processing {len(electrodes)} electrodes across {len(timepoints)} timepoints')
    
    # Single parallel call for all electrodes and timepoints
    sens_matrix, empirical_null_matrix = run_parallel_estimation_optimized(
        df,
        electrodes,
        design,
        metadata['subjectNumber'],
        timepoints,
        n_jobs=N_JOBS,
        backend=BACKEND,
        verbose=10
    )
    
    perc, p_val, z_score = estimate_stats(sens_matrix, empirical_null_matrix, two_sided=False)
    
    print(f"Sensitivity matrix shape: {sens_matrix.shape}")
    print(f"Empirical null matrix shape: {empirical_null_matrix.shape}")

    out = {
        'percentile' : perc,
        'p_val' : p_val,
        'z_score' : z_score
        }
    
    
    with open(out_pth/'sensor_reg_stats_all.pkl', 'wb') as f:
        pickle.dump(out, f)
