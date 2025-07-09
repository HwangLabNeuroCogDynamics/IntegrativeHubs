####################################################################
# Script to calculate EEG distance for ThalHi EEG data
####################################################################
import numpy as np
import pandas as pd
import mne
import os
from datetime import datetime
import scipy
from joblib import Parallel, delayed

def compute_inv_shrunk_covariance(x):
    #see http://www.diedrichsenlab.org/pubs/Walther_Neuroimage_2016.pdf
    t,n = x.shape #t measurements by channels
    #demean
    x = x - x.mean(0)
    #compute covariance
    sample = (1.0/t) * np.dot(np.transpose(x),x)
    #copute prior
    prior = np.diag(np.diag(sample))
    #compute shrinkage
    d = 1.0/n * np.linalg.norm(sample - prior,ord = 'fro')**2
    y = np.square(x)
    r2 = 1.0/n/t**2 * np.sum(np.sum(np.dot(np.transpose(y),y)))- \
    1.0/n/t*np.sum(np.sum(np.square(sample)))
    #compute the estimator
    shrinkage = max(0,min(1,r2/d))
    sigma = shrinkage*prior + (1-shrinkage)*sample
    #compute the inverse
    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            inv_sigma = np.linalg.inv(prior) #univariate
        else:
            raise
    
    return inv_sigma

def polynomial_kernel_distance(x1, x2):
    """
    Compute the distance between two patterns in the high-dimensional space
    induced by the polynomial kernel defined as:
    
        K(a, b) = (a · b')^2
    
    where a and b are row vectors. The corresponding distance is:
    
        distance(x1, x2) = sqrt( K(x1, x1) + K(x2, x2) - 2 * K(x1, x2) )
        
    Parameters
    ----------
    x1 : array_like of shape (n_features,)
        First pattern (row vector).
    x2 : array_like of shape (n_features,)
        Second pattern (row vector).
        
    Returns
    -------
    float
        The distance between x1 and x2 in the high-dimensional space.
    
    Example
    -------
    >>> x1 = np.array([1, 2, 3])
    >>> x2 = np.array([4, 5, 6])
    >>> dist = polynomial_kernel_distance(x1, x2)
    >>> print(dist)
    """
    # Compute the polynomial kernel values.
    # For vectors a and b, K(a, b) = (a dot b) ^ 2.
    K_x1x1 = np.dot(x1, x1) ** 2
    K_x2x2 = np.dot(x2, x2) ** 2
    K_x1x2 = np.dot(x1, x2) ** 2
    
    # Compute the squared distance.
    distance_sq = K_x1x1 + K_x2x2 - 2 * K_x1x2
    
    # Avoid numerical issues: ensure we don't take the sqrt of a negative number.
    distance_sq = max(distance_sq, 0)
    
    return np.sqrt(distance_sq)

ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'
sub = input()  
this_sub_path=ROOT+ 'eeg_preproc_RespToReviewers/' +str(sub)

all_probe = mne.read_epochs(this_sub_path+'/probe events-epo.fif')
all_probe.baseline = None
times = all_probe.times
picks = mne.pick_types(all_probe.info, meg=False, eeg=True, eog=False, stim=False)

evoked = all_probe.average() #calculate evoked response
resid  = all_probe.copy().subtract_evoked(evoked) #get residuals

#  Estimate the noise covariance from those residuals
noise_cov = mne.compute_covariance( resid, tmin=None, tmax=None, method="shrunk")

# Build the whitening operator
whitener = mne.cov.compute_whitener(noise_cov, resid.info, picks=picks, return_rank=False, return_colorer=False)
if isinstance(whitener, tuple):
    whitener, _ = whitener
    
epoch_data = all_probe.get_data()[:, picks, :] #trial by channels by timepoints
# whiten the data
channel_x_trials_corrected = np.einsum('ij, ejt -> eit', whitener, epoch_data) #this trials by channel by timepoints


### old approach to compute the inverse covariance for each epoch timepoint
# inv_sigma_epochs = np.zeros((64, 64, len(times)))
# # calcuate cov for each epoch timepoint
# for t in np.arange(len(times)):
#     inv_sigma_epochs[:,:, t] = scipy.linalg.fractional_matrix_power(compute_inv_shrunk_covariance(epoch_data[:,0:64,t]),0.5) 

# # normalize by noise cov 
# channel_x_trials_corrected = np.zeros((64,epoch_data.shape[0]  , len(times)))
# for t in np.arange(len(times)):
#     channel_x_trials = epoch_data[:,0:64, t].T
#     channel_x_trials_corrected[:,:, t] = np.dot(inv_sigma_epochs[:,:,t],channel_x_trials)


def compute_trial_corr(trial, times, channel_data):
    """
    Compute the correlation matrix for a single trial.
    
    Parameters:
        trial (int): trial index (will use trial and trial+1 for correlation)
        times (array-like): list or array of time indices
        channel_data (ndarray): assumed shape is (channels, trials, timepoints)
        
    Returns:
        coefs (ndarray): 2D array with shape (len(times), len(times)) containing correlations
    """
    n_times = len(times)
    coefs = np.zeros((len(range(0,n_times,10)), len(range(0,n_times,10)))) ## instead of full timepoints, only compute every 10th timepoint
    for it1,t1 in enumerate(range(0,n_times,10)):
        for it2,t2 in enumerate(range(0, n_times, 10)):
            x = channel_data[trial, :, t1]
            y = channel_data[trial - 1,:, t2] #distance from the previous trial
            #coefs[t1, t2] = polynomial_kernel_distance(x, y)
            r = np.corrcoef(x, y)[0, 1]
            if np.isnan(r):
                r = 0
            #convert to correlation distance
            coefs[it1, it2] = np.sqrt(2 * (1 - r))
    return coefs

n_trials = epoch_data.shape[0]
n_times = len(range(0,len(times),10))
# Prepare output matrix
coef_mat = np.zeros((n_trials, n_times, n_times))

# Parallelize the computation over trials:
results = Parallel(n_jobs=16)(
    delayed(compute_trial_corr)(trial, times, channel_x_trials_corrected)
    for trial in range(1, n_trials)) #starting from 2nd trial to compare with the previous one

# Fill in the computed correlation matrices:
for trial, corr_mat in enumerate(results):
    coef_mat[trial+1] = corr_mat

outfn = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_coef_mat.npy'
np.save(outfn, coef_mat)
print('Saved:', outfn)  # the end result is trial by time by time matrix of correlations


