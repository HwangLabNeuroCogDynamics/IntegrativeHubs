####################################################################
# Script to calculate EEG distance for ThalHi EEG data
####################################################################
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, cross_val_predict, KFold
from sklearn.model_selection import LeaveOneOut
from scipy.stats import zscore
from scipy.special import logit
from mne.time_frequency import tfr_morlet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datetime import datetime
import numpy as np
import pandas as pd
import mne
import os
import nibabel as nib
import glob
from datetime import datetime
import scipy
import nilearn
from nilearn.maskers import NiftiLabelsMasker
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

ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'
sub = input()  
this_sub_path=ROOT+ 'eeg_preproc_RespToReviewers/' +str(sub)

all_probe = mne.read_epochs(this_sub_path+'/probe events-epo.fif')
all_probe.baseline = None
times = all_probe.times
inv_sigma_epochs = np.zeros((64, 64, len(times)))
epoch_data = all_probe.get_data()

# calcuate cov for each epoch timepoint
for t in np.arange(len(times)):
    inv_sigma_epochs[:,:, t] = scipy.linalg.fractional_matrix_power(compute_inv_shrunk_covariance(epoch_data[:,0:64,t]),0.5) 

# normalize by noise cov 
channel_x_trials_corrected = np.zeros((64,epoch_data.shape[0]  , len(times)))
for t in np.arange(len(times)):
    channel_x_trials = epoch_data[:,0:64, t].T
    channel_x_trials_corrected[:,:, t] = np.dot(inv_sigma_epochs[:,:,t],channel_x_trials)



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
    coefs = np.zeros((n_times, n_times))
    for t1 in range(n_times):
        for t2 in range(n_times):
            x = channel_data[:, trial, t1]
            y = channel_data[:, trial + 1, t2]
            coefs[t1, t2] = np.corrcoef(x, y)[0, 1]
    return coefs

n_trials = epoch_data.shape[0] - 1
n_times = len(times)
# Prepare output matrix (note: last trial remains zeros, as the loop uses trial+1)
coef_mat = np.zeros((epoch_data.shape[0], n_times, n_times))

# Parallelize the computation over trials:
results = Parallel(n_jobs=24)(
    delayed(compute_trial_corr)(trial, times, channel_x_trials_corrected)
    for trial in range(n_trials))

# Fill in the computed correlation matrices:
for trial, corr_mat in enumerate(results):
    coef_mat[trial] = corr_mat

outfn = '/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_coef_mat.npy'
np.save(outfn, coef_mat)
print('Saved:', outfn)  # the end result is trial by time by time matrix of correlations


