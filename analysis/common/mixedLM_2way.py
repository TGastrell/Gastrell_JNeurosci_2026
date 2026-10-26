#%%
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from seaborn import despine
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.stats import false_discovery_control
from scipy.ndimage import gaussian_filter1d


def fit_mixedlm(df, formula, groups):
    model = MixedLM.from_formula(formula, groups=groups, data=df)
    result = model.fit(method='nm', maxiter=200)
    return result

def fit_timepoint(t, df_t):
    print(f'Fitting time {t}')
    if len(df_t.columns) == 5:
        formula = f'{df_t.columns[0]} ~ {df_t.columns[1]} * {df_t.columns[2]} + {df_t.columns[3]}'
    else:
        formula = f'{df_t.columns[0]} ~ {df_t.columns[1]} * {df_t.columns[2]}'
    groups = df_t.columns[-1]
    result = fit_mixedlm(df_t, formula, groups)
    return pd.DataFrame(result.params).T.assign(time=t), pd.DataFrame(result.pvalues).T.assign(time=t), pd.DataFrame(result.resid).assign(time=t)

def run_parallel_mixedlm(df, vars, n_jobs=10):
    '''
    Currently supports two way mixed effects linear model. 
    Follows convention: vars[0] ~ vars[1] * vars[2] + 1|vars[-1]
    '''
    results = Parallel(n_jobs=n_jobs, 
                         verbose=10,
                         batch_size=5)(delayed(fit_timepoint)(t, df.query('time == @t')[vars]) for t in df.time.unique())
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

def plot_fdr_results(all_pvals, dv='bias', legend = True):
    ''' Plot FDR-corrected p-values 
    
    Parameters
    ----------
    all_pvals : pd.DataFrame
        DataFrame with p-values for each time point and predictor.
    '''
    fdr_p = all_pvals.iloc[:, :-2].copy()
    fdr_p.iloc[:] = false_discovery_control(fdr_p, axis=0, method='bh')

    plt.figure(figsize=(10,2))
    for col in fdr_p.columns:
        plt.plot(all_pvals.time, gaussian_filter1d(-np.log(fdr_p[col], ), 4, 0), label=col)
    
    plt.axhline(-np.log(0.05), color='k', linestyle='--', label='p=0.05')
    plt.xlabel('Time (s)')
    plt.ylabel('-log(p)')
    plt.title(f'FDR-corrected p-values for {dv}')
    #plt.ylim(0,10)
    despine()
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_results_heatmap(results, pvals, dv='bias'):

    results_mat = results.iloc[:, :-2].copy().values
    pval_mat = pvals.iloc[:, :-2].copy().values

    fdr_p = false_discovery_control(pval_mat, axis=0, method='bh')
    significance_mask = fdr_p < 0.05
    results_sm = gaussian_filter1d(results_mat, 4, axis=0)
    results_sm[~significance_mask] = np.nan

    X = np.flip(results_sm.T, axis=0)
    centers = [results.time.min(), results.time.max(), 0, results.shape[1]-2]
    dy, = -np.diff(centers[2:])/(X.shape[0]-1)
    extent = [results.time.min(), results.time.max(), centers[2]+dy/2, centers[3]-dy/2]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title(f'Regression weights for {dv}')
    #ax.set_facecolor("#ebe5e5")
    image =  ax.imshow(X,
                aspect='auto',
                interpolation='none',
                origin='lower',
                extent=extent,
                cmap='coolwarm',
                vmin=-np.nanmax(np.abs(X)),
                vmax=np.nanmax(np.abs(X))
                )
    ax.set_yticks(np.arange(centers[3], centers[2]+dy, dy))
    ax.set_yticklabels(list(results.columns[:-2]))
    plt.colorbar(image, ax=ax, label=r'$\beta$')
    ax.set_xlabel('Time (s)')

# %%
