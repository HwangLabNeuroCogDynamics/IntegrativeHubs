import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

data = np.load('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/coefs/10001_whole_brain_rtcov_coef.npy')
M = data.mean(axis =0)
n_runs   = 8
run_size = M.shape[0] // n_runs   # 408/8 = 51
# reshape into (run₁, trial₁, run₂, trial₂)
M4 = M.reshape(n_runs, run_size,
               n_runs, run_size)

# average across run₁ and run₂ axes → (trial₁, trial₂) = (51, 51)
M_avg = M4.mean(axis=(0, 2))

print(M_avg.shape) 

plt.figure()
sns.heatmap(
    M_avg - M_avg.mean(),
    vmin=-0.05,      # set lower end of color scale
    vmax=0.05,       # set upper end
    center=0,     # ensures zero is in the middle of the colormap
    square=True,  # make cells square
    cbar_kws={'label': 'Value'}  # label for the color bar
)
plt.title('Trialwise Corr Coefficient Matrix for LSS', fontsize=20)
plt.tight_layout()
plt.show()

data = np.load('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/trialwiseRSA/coefs/10001_whole_brain_rtcov_coef.npy')
M = data.mean(axis =0)
n_runs   = 8
run_size = M.shape[0] // n_runs   # 408/8 = 51
# reshape into (run₁, trial₁, run₂, trial₂)
M4 = M.reshape(n_runs, run_size,
               n_runs, run_size)

# average across run₁ and run₂ axes → (trial₁, trial₂) = (51, 51)
M_avg = M4.mean(axis=(0, 2))

print(M_avg.shape) 
plt.figure()
sns.heatmap(
    M_avg - M_avg.mean(),
    vmin=-0.05,      # set lower end of color scale
    vmax=0.05,       # set upper end
    center=0,     # ensures zero is in the middle of the colormap
    square=True,  # make cells square
    cbar_kws={'label': 'Value'}  # label for the color bar
)
plt.title('Trialwise Corr Coefficient Matrix for GLMsingle', fontsize=20)
plt.tight_layout()
plt.show()