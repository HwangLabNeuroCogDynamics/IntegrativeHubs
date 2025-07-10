import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
import glob
import numpy as np
import pandas as pd
import mne
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ Config ------------------------
ROOT         = "/mnt/nfs/lss/lss_kahwang_hpc/ThalHi_data/"
BEHAVIOR_CSV = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/ThalHi_EEG_zRTs_full2.csv"
DIST_DIR     = os.path.join(ROOT, "RSA", "distance_results")
PATTERN      = os.path.join(DIST_DIR, "*_coef_mat.npy")
N_JOBS       = 16

# ------------------------ Load behavior ------------------------
fdf = pd.read_csv(BEHAVIOR_CSV)

def add_prev_match(df, var):
    """
    Adds two columns to df:
      - prev_<var>: the lag-1 value (grouped by sub+block)
      - <var>_repeat: 'same' or 'switch'
    based on df[var] equal to the shifted value.
    """
    df = df.sort_values(['sub','block','trial_n']).copy()
    grp = df.groupby(['sub','block'])[var]
    df['prev_'+var] = grp.shift(1)
    df[var+'_repeat'] = np.where(df[var] == df['prev_'+var], 'same', 'switch')
    return df

# ------------------------ Build subject‐wise data ------------------------
dfs, mats = [], []

for coef_file in sorted(glob.glob(PATTERN)):
    sub = int(os.path.basename(coef_file).split("_")[0])
    coef_mat = np.load(coef_file)  # shape (n_raw_trials, n_times)

    # 1) Prepare behavioral df for this subject
    df = (fdf.loc[fdf['sub'] == sub]
          .sort_values(['block','trial_n'])
          .reset_index(drop=True))
    df = add_prev_match(df, var='Subject_Respo')
    df['response_repeat'] = df['Subject_Respo_repeat']
    df = add_prev_match(df, var='Task')
    df['task_repeat']     = df['Task_repeat']
    df = add_prev_match(df, var='pic')
    df['probe_repeat']    = df['pic_repeat']
    df['Errors']          = (df['trial_Corr'] != 1).astype(int)
    for col in ["probe_repeat","response_repeat","task_repeat",
                "Trial_type","prev_target_feature_match"]:
        df[col] = df[col].astype("category")

    # 2) Figure out trial_vector from epochs metadata
    this_path = os.path.join(ROOT, 'eeg_preproc_RespToReviewers', str(sub))
    epochs    = mne.read_epochs(f"{this_path}/probe events-epo.fif", preload=False)
    meta_df   = (epochs.metadata
                 .sort_values(['block','trial_n'])
                 .reset_index(drop=True))
    meta_vec  = ((meta_df['block'] - 1) * 83 + meta_df['trial_n']).values
    df_vec    = ((df['block'] - 1) * 83 + df['trial_n']).values

    if len(df_vec) != len(meta_vec):
        trial_vector = np.intersect1d(df_vec, meta_vec).astype(int)
    else:
        trial_vector = np.arange(len(df_vec))

    if len(trial_vector) != coef_mat.shape[0]:
        print(f"Sub {sub}: trial_vector length " "≠ coef_mat rows")
        continue

    # 3) Subset only by trial_vector (no further mat deletion)
    df = df.loc[trial_vector].reset_index(drop=True)
    df['trial_vector'] = trial_vector
    # distances for this subject
    mats.append(coef_mat)
    dfs.append(df)

# concatenate across subjects
base_df   = pd.concat(dfs, ignore_index=True)
distances = np.vstack(mats)   # shape (n_trials_total, n_times)
n_times   = distances.shape[1]

# ------------------------ Model spec ------------------------
formula = (
    "ds ~ zRT + Errors "
    "+ C(Trial_type, Treatment(reference='IDS')) * (perceptual_change - perceptual_change.mean()) "
    "+ C(response_repeat) * C(task_repeat) "
    "+ C(prev_target_feature_match, Treatment(reference='switch_target_feature')) "
    "+ C(probe_repeat)"
)

# ------------------------ Fit mixed‐model per timepoint ------------------------
def fit_timepoint(t1):
    df = base_df.copy()
    # assign DV
    df['ds'] = distances[:, t1]
    # apply same trial_n & zRT filters as original per‐subject script
    mask = (df['trial_n'] != 0) & (df['zRT'].abs() <= 3)
    df = df.loc[mask].reset_index(drop=True)
    
    # drop block breaks
    # drop trials where to previous trial was dropped
    for i in range(1,len(df)):
        if df.loc[i,'trial_vector'] - df.loc[i-1, 'trial_vector'] != 1:
            df.loc[i, 'break'] = 1
    df = df.loc[df['break'] !=1]

    md  = smf.mixedlm(formula, df, groups=df['sub'], re_formula="~1")
    res = md.fit(method="lbfgs", disp=False)
    return res.tvalues

tvals_results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(fit_timepoint)(t) for t in range(n_times)
)

# assemble into a DataFrame (n_times × n_fixed_effects)
fixed_names = tvals_results[0].index.tolist()
tval_mat = pd.DataFrame(
    {name: [r[name] for r in tvals_results] for name in fixed_names},
    index = np.arange(n_times)
)

# ------------------------ Plot t-values ------------------------
plt.figure(figsize=(8,3))
# example: plot t-values of 'perceptual_change_c' over time
sns.lineplot(x=epochs.times, y=tval_mat["zRT"], label="zRT")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Timepoint')
plt.ylabel('t value')
plt.title('Mixed-effects t-stat for zRT over time')
plt.tight_layout()
plt.show()

