import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import vonmises
from scipy.special import iv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def wrapTopi(theta):
    '''
    Wrap array of angles in pi radians from -pi to pi
    Params:
    theta: array of angles in pi radians
    Returns:
    wrapped thetas
    '''
    return (theta + np.pi) % (2 * np.pi) - np.pi

def wrapTo2pi(theta):
    '''
    Wrap array of angles in pi radians from 0 to 2pi
    Params:
    theta: array of angles in pi radians
    Returns:
    wrapped thetas
    '''
    return (theta + 2 * np.pi) % (2 * np.pi)

def k2sd(K):
    '''
    Convert kappa parameter to circular standard deviation
    Params:
    K - kappa of von Mises distribution
    Returns:
    S - standard deviation of circular variable
    '''
    if K == 0:
        S = np.Inf
    elif np.isinf(K):
        S = 0
    else:
        S = np.sqrt(-2*np.log(iv(1,K)/iv(0,K)))
    return S

def sd2k(S):
    '''
    Convert circular standard deviation to kappa parameter
    Params:
    S - standard deviation of circular variable
    Returns:
    K - kappa of von Mises distribution
    '''
    R = np.exp(-S**2/2.)
    if R < .85:
        K = -.4 + 1.39*R + .43/(1 - R)
    elif R < .53:
        K = 2*R + R**3 + (5*R**5)/6.
    else:
        K = 1./(R**3 - 4*R**2 + 3*R)
    return(K)


def circular_posterior(mu_0, kappa_0, mu_d, kappa_d):
    # Convert mean directions to complex numbers
    z_0 = kappa_0 * np.exp(1j * mu_0)
    z_d =  kappa_d * np.exp(1j * mu_d)

    # Compute the posterior mean direction in complex form
    z_post = z_0 + z_d

    # Convert the complex number back to an angle
    mu_post = np.angle(z_post)  # Argument of the complex number
    kappa_post = np.abs(z_post)
    
    return mu_post, kappa_post

def circular_difference(theta_a, theta_b):

    # Convert directions to complex numbers
    z_a = np.exp(1j * theta_a)
    z_b = np.exp(1j * theta_b)

    z_dif = z_a/z_b

    theta_dif = np.angle(z_dif)

    return theta_dif


def plot_vonmises(x, hist = False, showMu = False, **kwargs):
    # Fit vonmises distribution to x and plot it. Useful to pass this to a facetGrid function
    hist_color = kwargs.get('hist_color', 'skyblue')
    line_color = kwargs.get('line_color', 'skyblue')

    ax = plt.gca()
    if hist:
        sns.histplot(x, stat='density',ax=ax, color=hist_color, binwidth=0.2)
    
    # Fit Von Mises distribution
    kappa, mu, scale = vonmises.fit(x)
    
    # Create values for the fitted curve
    x_vals = np.linspace(-np.pi, np.pi, 500)
    fitted_pdf = vonmises.pdf(x_vals, kappa, mu, scale)
    
    # Plot the fitted curve
    ax.plot(x_vals, fitted_pdf, color=line_color)
    if showMu:
        plt.plot([mu,mu], [0, 2], linewidth = 2, color = line_color, linestyle = 'dashed')


def cumulative_prior(stimDir, stimKappa, blockNr, PID, reset_per_block=True):
    x=pd.DataFrame()
    for i in range(len(stimDir)):
        if i == 0 :
            y = pd.DataFrame(dict(mu = [0], kappa = [0])) # Initialise
        elif PID[i] != PID[i-1]:
            y = pd.DataFrame(dict(mu = [0], kappa = [0])) # new PID new prior
        elif blockNr[i] != blockNr[i-1] :
            if reset_per_block : # First instance of a new block reset prior
                y = pd.DataFrame(dict(mu = [0], kappa = [0]))
        else :
            m, k = circular_posterior(x.loc[i-1]['mu'], x.loc[i-1]['kappa'], stimDir[i-1], stimKappa[i-1])
            y = pd.DataFrame(dict(mu = [m], kappa = [k]))

        x = pd.concat([x, y]).reset_index(drop=True)

    return x.mu, x.kappa

def window_prior(stimDir, stimKappa, trialNrInBlock, window_size):
    x=pd.DataFrame()

    for i in range(len(stimDir)) : # for each trial
        if trialNrInBlock[i]-1 < window_size :
            window = pd.DataFrame(dict(mu=[0], kappa=[0]))
        else :
            window = pd.DataFrame()
            for j in range(-window_size, 0) : # collect information starting window_size trials back and start cumulatively building prior window
                if j == -window_size: # first trial in window has no prior
                    z = pd.DataFrame(dict(mu=[stimDir[i+j]], kappa=[stimKappa[i+j]]))
                else :
                    mz, kz = circular_posterior(window.loc[window_size+j-1]['mu'], window.loc[window_size+j-1]['kappa'], stimDir[i+j], stimKappa[i+j]) # integrate previous posterior with current direction
                    z = pd.DataFrame(dict(mu=[mz], kappa=[kz]))

                window = pd.concat([window, z]).reset_index(drop=True) # add z to window
                
        x = pd.concat([x, window.tail(1)]).reset_index(drop=True) # take the last entry in the window as the prior for the current trial i

    return x.mu, x.kappa

def cmean(theta):
    theta = np.array(theta)
    if np.any(np.abs(theta) > np.pi): 
        print('The range of values must be between -pi and pi')
        return np.nan
    else:
        thetaHat = np.arctan2(np.sin(theta).sum(),np.cos(theta).sum())
        return thetaHat
    
def cstd(theta):
    theta = np.array(theta)
    if np.any(np.abs(theta) > np.pi): 
        print('The range of values must be between -pi and pi')
        return np.nan
    else:
        R = np.sqrt(np.sin(theta).sum()**2+np.cos(theta).sum()**2)/theta.size
        thetaSD = np.sqrt(-2*np.log(R))    
        return thetaSD   

def circular_sum(theta_a, theta_b):
    '''
    Calculate the angular sum between two angles in radians
    Params:
    theta_a - angle on the lhs of the sum in radians
    theta_b - anfle on the rhs of the sum in radians
    Returns:
    theta_sum - sum of theta_a and theta_b in radians
    '''
    # Convert directions to complex numbers
    z_a = np.exp(1j * theta_a)
    z_b = np.exp(1j * theta_b)

    z_dif = z_a*z_b

    theta_sum = np.angle(z_dif)

    return theta_sum

def cmplxOLS(pred, crit):
    '''
    Compute complex-valued OLS regression
    **Input**
    pred: predictor structure (N trials x M predictors) in pi radians
    crit: criterion structure (N trial x J criteria) in pi radians
    '''
    # transform values from real to complex
    pred = np.exp(pred * 1j)
    crit = np.exp(crit * 1j)
    # reformat shape if necessary
    if len(pred.shape) == 1:
        pred = pred.reshape(-1, 1)
    if len(crit.shape) == 1:
        crit = crit.reshape(-1, 1)
    # test if the N trials is the same
    if pred.shape[0] != crit.shape[0]:
        raise ValueError('Mismatch betweeh input shapes!') 
    coefs = (np.asmatrix(
        np.asmatrix(pred).H
        * np.asmatrix(pred)
    ).I * (
        np.asmatrix(pred).H
        * np.asmatrix(crit)
    ))
    return (np.abs(coefs), np.angle(coefs))

def get_signed_error(responseError, priorDist):
    '''
    Compute Signed Errors
    '''
    # Computed Signed errors such that errors are positive when they are towards the prior.
    idx_swap_sign = ((np.sign(responseError) == np.sign(priorDist)) & (responseError < 0)) | ((np.sign(responseError) != np.sign(priorDist)) & (responseError > 0))
    signed_error = responseError.copy()
    signed_error[idx_swap_sign] *= -1
    return signed_error

def resultant_length(angles):
    """
    Calculate resultant length for circular data.
    
    Parameters:
    angles: array-like of angles in radians
    
    Returns:
    float: resultant length (0 to 1)
    """
    n = len(angles)
    if n == 0:
        return 0
    
    # Sum of unit vectors
    sum_cos = np.sum(np.cos(angles))
    sum_sin = np.sum(np.sin(angles))
    
    # Magnitude of resultant vector, normalized by n
    R = np.sqrt(sum_cos**2 + sum_sin**2) / n
    
    return R

def circular_confidence_interval(angles, confidence_level=0.95):
    """
    Calculate confidence interval for circular mean.
    
    Parameters:
    angles: array-like of angles in radians
    confidence_level: float, confidence level (default 0.95)
    
    Returns:
    tuple: (lower_bound, upper_bound) in radians, or (NaN, NaN) if data too dispersed
    """
    import numpy as np
    from scipy import stats
    
    n = len(angles)
    if n < 2:
        return np.nan, np.nan
    
    # Calculate circular mean and resultant length
    sum_cos = np.sum(np.cos(angles))
    sum_sin = np.sum(np.sin(angles))
    R = np.sqrt(sum_cos**2 + sum_sin**2) / n
    mean_angle = np.arctan2(sum_sin, sum_cos)
    
    # If data too dispersed, CI is meaningless
    if R < 0.1:
        return np.nan, np.nan
    
    # Estimate concentration parameter (kappa) from R
    if R < 0.53:
        kappa = 2*R + R**3 + (5*R**5)/6
    elif R >= 0.85:
        kappa = 1/(R**3 - 4*R**2 + 3*R)
    else:
        kappa = -0.4 + 1.39*R + 0.43/(1-R)
    
    # Calculate confidence interval half-width
    alpha = 1 - confidence_level
    chi2_val = stats.chi2.ppf(confidence_level, df=1)
    
    # Approximate CI half-width
    if kappa > 2:
        d = np.sqrt(chi2_val / (n * kappa))
    else:
        # For low concentration, use different approximation
        d = np.sqrt(-2 * np.log(alpha/2) / (n * R**2))
    
    # Ensure d is reasonable
    d = min(d, np.pi)
    
    # Calculate bounds
    lower = mean_angle - d
    upper = mean_angle + d
    
    # Wrap to [-π, π]
    lower = np.arctan2(np.sin(lower), np.cos(lower))
    upper = np.arctan2(np.sin(upper), np.cos(upper))
    
    return lower, upper