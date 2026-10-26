from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

def fit_mixedlm(df, formula, groups):
    model = MixedLM.from_formula(formula, groups=groups, data=df)
    result = model.fit(method='nm', maxiter=200)
    return result

def fit_timepoint(t, df_t, dv='cpp'):
    print(f'Fitting time {t}')
    formula = f'{dv} ~ Prior_Precision'
    groups = df_t['subjectNumber']
    result = fit_mixedlm(df_t, formula, groups)
    return pd.DataFrame(result.params).T.assign(time=t), pd.DataFrame(result.pvalues).T.assign(time=t), pd.DataFrame(result.resid).assign(time=t)

def run_parallel_mixedlm(df, dv, n_jobs=10):
    results = Parallel(n_jobs=n_jobs, 
                         verbose=10,
                         batch_size=5)(delayed(fit_timepoint)(t, df.query('time == @t')[[dv,'Prior_Precision', 'subjectNumber']], dv=dv) for t in df.time.unique())
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