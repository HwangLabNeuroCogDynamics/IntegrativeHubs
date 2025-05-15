import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub'] #this is usable subjects after Steph's filtering.
M = np.zeros((len(subjects), 408, 408))
for i, sub in enumerate(subjects):
    data = np.load('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/coefs/%s_whole_brain_rtcov_coef.npy' %sub)
    if data.shape[1] == 408:
        M[i,:,:] = np.nanmean(data, axis=0)
M_avg = np.nanmean(M, axis=0)
print(M_avg.shape) 

D = np.zeros((len(subjects), 408, 408))
for i, sub in enumerate(subjects):
    data = np.load('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/trialwiseRSA/rdms_correlation_whitened_betas/%s_Schaefer400_Morel_BG_rdm_corr_whitened.npy' %sub)
    if data.shape[1] == 408:
         D[i,:, :] = np.nanmean(data, axis=0)

D_avg = np.nanmean(D, axis=0)


plt.figure()
sns.heatmap(
    M_avg,
    vmin=-0.4,      
    vmax=0.4,       
    center=0,     
    square=True,  
    cmap = "vlag",
)
plt.title('Trialwise Corr Coefficient Matrix for LSS', fontsize=20)
plt.tight_layout()
plt.show()


print(M_avg.shape) 
plt.figure()
sns.heatmap(
    D_avg ,
    vmin=-0.4,      
    vmax=0.4,       
    center=0,     # ensures zero is in the middle of the colormap
    square=True,  
    cmap = "vlag",
)
plt.title('Trialwise Corr Coefficient Matrix for GLMsingle', fontsize=20)
plt.tight_layout()
plt.show()