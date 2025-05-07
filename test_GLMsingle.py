#testing GLMSingle
# to install, on thalamege, do pip install git+https://github.com/cvnlab/GLMsingle.git
# to install on argon, first "source activate" into your env, then do "pip install git+https://github.com/cvnlab/GLMsingle.git"
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib
import os
from os.path import join, exists, split
import time
import urllib.request
import warnings
from tqdm import tqdm
from pprint import pprint
warnings.filterwarnings('ignore')
import matplotlib 
from glmsingle.glmsingle import GLM_single
import subprocess
from nilearn.input_data import NiftiMasker

# variables that will contain bold time-series and design matrices from each run
data = [] # list of numpy arrays, each entry is a run, and the run is a 4D numpy array of x, y, z, t
design = [] # list of numpy arrays, each entry is a run, and the run is a 2D numpy array of t, n, with n being the number of regressors or conditions.
sub = 10037
n_runs = 8
TR = 1.8
n_timepts = 1704 # this is after removing 3TRs per subject, else 1728. I assume this is the same for all subjects?
deconv_dir = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-%s" % sub
data_dir = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/fmriprep/sub-%s/func/" % sub
#run_breaks = [0, 216, 432, 648, 864, 1080, 1296, 1512, 1728] # these are the time points for each run, assume this is the same for all subjects??
run_breaks = np.arange(0, 1728, 213) # each run is 213, after removing 3TRs from 216
concat_str = '1D: ' + ' '.join(str(i) for i in run_breaks[0:8])

### use AFNI 3dDeconvolve to create the design matrix
# note this part is calling on AFNI in python, this is not a python function.
# it is easy to do this on thalamege, but on argon AFNI needs to be run in container so I am not sure how to do this.
# one solution is to just write a bash script and run this part separately, should be easy and fast to do.
# Note here not reating different cue types as different conditions, don't think its necessary when the goal is to get the GLM coefficients for each trial.  
# xmat_file = os.path.join(deconv_dir, "cueonset_IM.xmat.1D")
# cmd = [
#     "3dDeconvolve",
#       "-nodata", str(n_timepts), str(TR),
#       "-polort", "-1",
#       "-concat", concat_str,
#       "-num_stimts", "1",
#       "-local_times",
#       "-stim_times", "1",
#         os.path.join(deconv_dir, "cue_stimtimes.1D"),
#         "TENT(0.6,2.4,2)",
#       "-stim_label", "1", "cue",
#       "-x1D", xmat_file,
#       "-x1D_stop",
#       "-allzero_OK"
# ]
# print("Running 3dDeconvolve:"," ".join(cmd))
# subprocess.run(cmd, check=True)

# load the full design‐matrix, extract first "tent", binarize, and save
#mat = np.loadtxt(xmat_file)      # comments auto‐ignored
#onsets = ( mat[:, 0] != 0).astype(int)   # convert on set to binary, will loose some termporal percision without AFNI's internal up then down sampling of full design matrix.
# see: https://glmsingle.readthedocs.io/en/latest/wiki.html#my-experiment-design-is-not-quite-synchronized-with-my-fmri-data 

# need to create mask to improve efficiency of GLMsingle
# Annoyingly masks are in AFNI format and nilearn / nibabel use NIFTI format
input_afni   = os.path.join(deconv_dir, "combined_mask+tlrc")
output_nifti = os.path.join(deconv_dir, "combined_mask.nii.gz")
cmd = [ "3dAFNItoNIFTI", "-prefix", output_nifti, input_afni]
print("Running command:", " ".join(cmd))
subprocess.run(cmd, check=True)
mask_file = os.path.join(deconv_dir, "combined_mask.nii.gz")
masker = NiftiMasker(mask_img=mask_file, standardize=False, detrend = False)  

# organize the design matrix and data into a list of runs given GLMsingle's required format
# read stim_times file and do the rounding ourselves. Faster and give same results anyway.
mat = np.loadtxt(os.path.join(deconv_dir, "cue_stimtimes.1D"))

for run in range (n_runs):
    fn = "sub-%s_task-ThalHi_run-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" %(sub, run+1)
    bold_img = os.path.join(data_dir, fn)
    ts = masker.fit_transform(bold_img).T
    data.append(ts[:, 3:]) #drop first 3 TRs for each run
    # I am just using "one" condition, the shape expects [n_timepts, #cond], so I had to add the "newaxis" to make it 2D. My guess is this is a quirk from matlab...
    #design.append(onsets[run_breaks[run]:run_breaks[run+1], np.newaxis])  
    onsets = np.zeros(213) #216 - 3, 213 TRs per run
    onsets[np.round((mat[run] + 0.6)/TR).astype(int)]=1 #because we deleted the first 3 runs, shift by 0.6 as there was a 6 sec delay (6-3*1.8 = 0.6)
    design.append(onsets[:, np.newaxis])

# create a directory for saving GLMsingle outputs. It appears that they have no option of saving "subject name" to the fn.... 
outputdir_glmsingle = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/"
#os.makedirs(outputdir_glmsingle,exist_ok=True)

start_time = time.time()
opt = dict()
# set important fields for completeness (but these would be enabled by default)
opt['wantlibrary'] = 1 #HRF library
opt['wantglmdenoise'] = 1 #learn noise components for denoising
opt['wantfracridge'] = 1 #ridge regression
opt['wantfileoutputs'] = [0,0,0,0] #don't save any outputs using their format because I cant read it
opt['wantmemoryoutputs'] = [1,1,1,1] #use memory outputs instead
glmsingle_obj = GLM_single(opt)
stimdur = 0.5 #cue + probe, ended around RT
tr = 1.8 
results_glmsingle = glmsingle_obj.fit(design, data, stimdur, tr, outputdir=outputdir_glmsingle)  

elapsed_time = time.time() - start_time
print( '\telapsed time: ', f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}' )


### whats in the output?
# R2. how well the dodel fit
vol_img = masker.inverse_transform(np.squeeze(results_glmsingle['typed']['R2']))
vol_img.to_filename(os.path.join(outputdir_glmsingle, "%s_R2.nii.gz" %sub))

vol_img = masker.inverse_transform(np.squeeze(results_glmsingle['typed']['noisepool']))
vol_img.to_filename(os.path.join(outputdir_glmsingle, "%s_noisevoxels.nii.gz" %sub))

vol_img = masker.inverse_transform(np.squeeze(results_glmsingle['typed']['HRFindex']))
vol_img.to_filename(os.path.join(outputdir_glmsingle, "%s_HRFindex.nii.gz" %sub))

vol_img = masker.inverse_transform(np.squeeze(results_glmsingle['typed']['betasmd']).mean(axis=1))
vol_img.to_filename(os.path.join(outputdir_glmsingle, "%s_meanBeta.nii.gz" %sub))

vol_img = masker.inverse_transform(np.squeeze(results_glmsingle['typed']['betasmd']).T)
vol_img.to_filename(os.path.join(outputdir_glmsingle, "%s_TrialBetas.nii.gz" %sub))