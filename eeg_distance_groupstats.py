# group stats for EEG distance results
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label
from scipy import stats
from joblib import Parallel, delayed

#permutation function
def _one_perm_mass(dM, threshold, ttest):
    # pick random ±1 signs for each subject, broadcast to (sub, x, y)
    signs = (2*np.random.randint(0,2,size=dM.shape[0]) - 1)[:,None,None]
    perm  = dM * signs

    # compute either t‐map or mean‐map
    if ttest:
        mat = stats.ttest_1samp(perm, 0, axis=0)[0]
    else:
        mat = perm.mean(axis=0)

    # find largest absolute cluster‐mass
    best = 0
    # positive clusters
    cmat, ncl = label(mat > threshold)
    for c in range(1, ncl+1):
        mass = mat[cmat==c].sum()
        if mass > best: best = mass
    # negative clusters
    cmat, ncl = label(mat < -threshold)
    for c in range(1, ncl+1):
        mass = -mat[cmat==c].sum()
        if mass > best: best = mass

    return best

def matrix_permutation(M1, M2, threshold, p, ttest=True, n_jobs=16):
    """ cluster‐mass permutation test, parallelized over iterations """
    if M1.shape != M2.shape:
        raise ValueError("M1 and M2 must have same shape")
    dM = M1 - M2

    # 1) compute null distribution in parallel
    null_mass = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_perm_mass)(dM, threshold, ttest)
        for _ in range(5000)
    )
    null_mass = np.array(null_mass, float)

    # 2) get cluster‐mass thresholds
    pos_thr = np.quantile(null_mass, 1-0.5*p)
    neg_thr = np.quantile(null_mass, 0.5*p)

    # 3) compute observed statistic
    if ttest:
        obs = stats.ttest_1samp(dM, 0, axis=0)[0]
    else:
        obs = dM.mean(axis=0)

    # 4) form mask of significant clusters
    mask = np.zeros_like(obs, bool)
    # positive
    cmat, ncl = label(obs > threshold)
    for c in range(1, ncl+1):
        if obs[cmat==c].sum() > pos_thr:
            mask[cmat==c] = True
    # negative
    cmat, ncl = label(obs < -threshold)
    for c in range(1, ncl+1):
        if (-obs[cmat==c]).sum() > neg_thr:
            mask[cmat==c] = True

    return mask, obs

# ------------------------ Config ------------------------
dataset_dir = '/mnt/nfs/lss/lss_kahwang_hpc/ThalHi_data/RSA/distance_results'
files = sorted(glob.glob(os.path.join(dataset_dir, "*_model_betas_results.npz")))
n_sub = len(files)

# 1) Load just the first file to get parameter names
sample = np.load(files[0])
params = sample.files
t1, t2 = sample[params[0]].shape
sample.close()

# 2) Pre-allocate a dict of lists
all_data = {p: [] for p in params}

# 3) Loop over subjects and collect each param’s 2D map
for fn in files:
    data = np.load(fn)
    try:
        for p in params:
            all_data[p].append(data[p].astype(np.float32))
    except Exception as e:
        print(f"Error processing {fn}: {e}")
    data.close()

# 4) Stack into arrays of shape (n_sub, t1, t2)
group_data = {p: np.stack(all_data[p], axis=0) for p in params}

# 5) Run permutation test and plot for each parameter
threshold = 2.5   # cluster‐forming t‐threshold
p_val     = 0.05

out_fig_dir = os.path.join(dataset_dir, "figures")
os.makedirs(out_fig_dir, exist_ok=True)

for p, M in group_data.items():
    zeros = np.zeros_like(M)
    mask, tmap = matrix_permutation(M, zeros, threshold, p_val, ttest=True)

    # 6) Plot heatmap of the group t‐map
    plt.figure(figsize=(6,5))
    sns.heatmap(tmap*mask, cmap="RdBu_r", center=0,
                cbar_kws={"label":"t value"})
    plt.title(f"Group t‐map: {p}")
    plt.xlabel("Trial N timepoint")
    plt.ylabel("Trial N-1 timepoint")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_fig_dir, f"tmap_{p}.png"))
    plt.close()


### if 1D
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label
from scipy import stats
from joblib import Parallel, delayed

# ---------------- permutation (unchanged) ----------------
def _one_perm_mass(dM, threshold, ttest):
    signs = (2*np.random.randint(0,2,size=dM.shape[0]) - 1)[:,None]
    perm  = dM * signs
    if ttest:
        mat = stats.ttest_1samp(perm, 0, axis=0)[0]
    else:
        mat = perm.mean(axis=0)
    best = 0
    cmat, ncl = label(mat > threshold)
    for c in range(1, ncl+1):
        m = mat[cmat==c].sum()
        if m>best: best = m
    cmat, ncl = label(mat < -threshold)
    for c in range(1, ncl+1):
        m = -mat[cmat==c].sum()
        if m>best: best = m
    return best

def matrix_permutation(M1, M2, threshold, p, ttest=True, n_jobs=16):
    if M1.shape != M2.shape:
        raise ValueError("M1 and M2 must have same shape")
    dM = M1 - M2
    # null distribution
    null_mass = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_perm_mass)(dM, threshold, ttest)
        for _ in range(5000)
    )
    null_mass = np.array(null_mass, float)
    pos_thr = np.quantile(null_mass, 1-0.5*p)
    neg_thr = np.quantile(null_mass,  0.5*p)
    # observed statistic
    if ttest:
        obs = stats.ttest_1samp(dM, 0, axis=0)[0]
    else:
        obs = dM.mean(axis=0)
    # cluster‐mask
    mask = np.zeros_like(obs, bool)
    cmat, ncl = label(obs > threshold)
    for c in range(1,ncl+1):
        if obs[cmat==c].sum() > pos_thr:
            mask[cmat==c] = True
    cmat, ncl = label(obs < -threshold)
    for c in range(1,ncl+1):
        if (-obs[cmat==c]).sum() > neg_thr:
            mask[cmat==c] = True
    return mask, obs

# ------------------------ Config ------------------------
dataset_dir = '/mnt/nfs/lss/lss_kahwang_hpc/ThalHi_data/RSA/distance_results'
files = sorted(glob.glob(os.path.join(dataset_dir, "*_model_betas_results.npz")))
n_sub = len(files)
if n_sub == 0:
    raise RuntimeError("No files found!")

# 1) peek at the first file to get regressor names + length L
sample = np.load(files[0])
params = sample.files                       # e.g. ['Intercept', 'dslope', '...']
L      = sample[params[0]].shape[0]         # the 1D length
sample.close()

# 2) collect each subject's 1D map in a dict of lists
all_data = {p: [] for p in params}
for fn in files:
    data = np.load(fn)
    try:
        for p in params:
            all_data[p].append(data[p].astype(np.float32))
    except Exception as e:
        print(f"Error processing {fn}: {e}")
    data.close()

# 3) stack into (n_sub, L) arrays
group_data = {p: np.stack(all_data[p], axis=0) for p in params}

# 4) run permutation and plot each regressor
threshold = 2   # cluster‐forming threshold on t
p_val     = 0.05
out_fig   = os.path.join(dataset_dir, "figures_1d")
os.makedirs(out_fig, exist_ok=True)

for p, M in group_data.items():
    zeros = np.zeros_like(M)
    mask, tmap = matrix_permutation(M, zeros, threshold, p_val, ttest=True)

    # 5) line‐plot of t‐map with cluster mask
    idx = np.arange(L)
    plt.figure(figsize=(8,3))
    plt.plot(idx, tmap, label="t‐stat", color="black")
    # highlight significant clusters
    sig = np.where(mask)[0]
    plt.scatter(sig, tmap[sig], color="red", s=2, label="cluster‐signif")
    plt.axhline(threshold,  color="gray", linestyle="--")
    plt.axhline(-threshold, color="gray", linestyle="--")
    plt.title(f"Group t‐map: {p}")
    plt.xlabel("Sample index (originally time or feature)")
    plt.ylabel("t value")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


    plt.savefig(os.path.join(out_fig, f"ttmap_1d_{p}.png"))
    plt.close()
