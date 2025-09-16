#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ActFlow on THalHi Switch cost analysis.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import json

subject = input()

TIMESERIES_PATH = '/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-%s/errts.nii.gz' %subject
BETAS_PATH  = '/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/sub-%s/%s_TrialBetas_resRT.nii.gz' %(subject, subject)
ROI_MASK_PATH = '/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/ROIs/Morel_Striatum_Yeo400.nii.gz'
OUTPUT_DIR = '/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/actflow'
SOURCE_ROI_IDS = [330, 331, 332]
TARGET_ROI_IDS = None

# Default CV, will run the other half later
SPLIT_CONN = "second"   # 'first' or 'second' (timeseries connectivity)
SPLIT_PRED = "first"    # 'first' or 'second' (betas prediction)
PCA_K = 40
N_JOBS = 8  # threads; keep BLAS threads = 1 (set above)
# Placeholder global used by the function
SOURCE_ROI = None

# =========================
# need to inflate .nii.gz → .nii inline so we can map it in memory
# =========================
print(" Ensuring uncompressed .nii for true memory-mapping...")

ts_p = Path(TIMESERIES_PATH)
if "".join(ts_p.suffixes[-2:]) == ".nii.gz":
    ts_out = ts_p.with_suffix("").with_suffix(".nii")
    if not ts_out.exists():
        print(f"         Inflating {ts_p} -> {ts_out}")
        _img = nib.load(str(ts_p))
        nib.save(_img, str(ts_out))
    TIMESERIES_PATH = str(ts_out)

be_p = Path(BETAS_PATH)
if "".join(be_p.suffixes[-2:]) == ".nii.gz":
    be_out = be_p.with_suffix("").with_suffix(".nii")
    if not be_out.exists():
        print(f"         Inflating {be_p} -> {be_out}")
        _img = nib.load(str(be_p))
        nib.save(_img, str(be_out))
    BETAS_PATH = str(be_out)

print(f" Timeseries mmappable path: {TIMESERIES_PATH}")
print(f" Betas mmappable path:      {BETAS_PATH}")

ts_img = nib.load(TIMESERIES_PATH, mmap=True)
be_img = nib.load(BETAS_PATH,      mmap=True)
TS_PROXY   = ts_img.dataobj
BETA_PROXY = be_img.dataobj

print("Resampling ROI mask to timeseries space (nearest-neighbor)…")

# Load original mask (whatever its grid/affine is)
mask_img_orig = nib.load(ROI_MASK_PATH)

# Use nilearn's resample_to_img with nearest neighbor to preserve label integers
from nilearn.image import resample_to_img
mask_img_rs = resample_to_img(
    source_img=mask_img_orig,
    target_img=ts_img,                 # resample mask to the timeseries grid
    interpolation="nearest"            # CRITICAL: preserves discrete labels
)

# Convert to integer array (nearest gives floats but with exact label values)
ROI_MASK = np.asanyarray(mask_img_rs.get_fdata(), dtype=np.float32)
# If any NaNs sneak in at edges, map them to background label 0
if np.isnan(ROI_MASK).any():
    ROI_MASK = np.nan_to_num(ROI_MASK, nan=0.0)

ROI_MASK = ROI_MASK.astype(np.int32, copy=False)

print(f" Resampled mask shape: {ROI_MASK.shape} (should match TS proxy spatial dims {TS_PROXY.shape[:3]})")
if ROI_MASK.shape != TS_PROXY.shape[:3]:
    raise RuntimeError(f"Resampled mask mismatch: {ROI_MASK.shape} vs TS {TS_PROXY.shape[:3]}")

out_mask_path = Path(OUTPUT_DIR) / "roi_mask_resampled_to_ts.nii.gz"
out_mask_path.parent.mkdir(parents=True, exist_ok=True)
nib.save(nib.Nifti1Image(ROI_MASK.astype(np.int16, copy=False), ts_img.affine), str(out_mask_path))
print(f"Saved resampled mask -> {out_mask_path}")

print(f" TS shape:  {TS_PROXY.shape}  (X,Y,Z,T)")
print(f" BET shape: {BETA_PROXY.shape} (X,Y,Z,N)")
print(f" Mask shape:{ROI_MASK.shape}   (X,Y,Z) [int labels]")

# ===== derive ROI ids once ( will exclude source per loop) =====
ALL_ROI_IDS = sorted(int(i) for i in np.unique(ROI_MASK) if int(i) != 0)
if not ALL_ROI_IDS:
    raise SystemExit("No target ROI ids found. Check ROI mask.")

# =========================
# parallel actflow per target 
# =========================
def predict_target_roi(target_roi_id: int) -> dict:
    """
    returns trialwise Pearson r and split-half noise-ceiling-normalized r.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    block_ts_conn = np.asarray(TS_PROXY[..., CONN_SLICE],  dtype=np.float32)   # (X,Y,Z,Tc)
    block_be_pred = np.asarray(BETA_PROXY[..., PRED_SLICE],dtype=np.float32)   # (X,Y,Z,Np)

    src_mask = (ROI_MASK == int(SOURCE_ROI))
    tgt_mask = (ROI_MASK == int(target_roi_id))

    X_conn = block_ts_conn[src_mask, :].T   # (Tc, p)
    Y_conn = block_ts_conn[tgt_mask, :].T   # (Tc, q)
    X_pred = block_be_pred[src_mask, :].T   # (Np, p)
    Y_true = block_be_pred[tgt_mask, :].T   # (Np, q)
    N_trials, q = Y_true.shape

    # ----- Fit connectivity on the connectivity half -----
    Xc = StandardScaler().fit_transform(X_conn)
    pca = PCA(n_components=int(PCA_K), svd_solver="auto", random_state=0)
    Z = pca.fit_transform(Xc)
    reg = LinearRegression()
    reg.fit(Z, Y_conn)
    B = reg.coef_.T                        # (k, q)
    P = pca.components_.T                  # (p, k)
    W = P @ B                              # (p, q)

    # ----- Predict held-out betas -----
    Xp = StandardScaler().fit_transform(X_pred)
    Y_hat = Xp @ W                              # (Np, q)

    # ----- Trialwise Pearson r (across voxels) -----
    Y_true_c = Y_true - Y_true.mean(axis=1, keepdims=True)
    Y_hat_c  = Y_hat  - Y_hat.mean(axis=1, keepdims=True)
    num   = np.sum(Y_true_c * Y_hat_c, axis=1)
    den_y = np.sqrt(np.sum(Y_true_c**2, axis=1))
    den_h = np.sqrt(np.sum(Y_hat_c**2,  axis=1))
    den   = den_y * den_h
    trial_r = np.where(den > 1e-12, num / den, np.nan).astype(np.float32)

    # Split-half noise ceiling r_nc from Y_true (first half vs second half), Xitong's method
    mid = N_trials // 2
    if mid == 0:
        r_nc = np.nan
    else:
        A_idx = slice(0,  mid)
        B_idx = slice(mid, N_trials)

        # Means across trials in each half, then mean-center across voxels
        mu_A = Y_true[A_idx].mean(axis=0)
        mu_B = Y_true[B_idx].mean(axis=0)
        mu_A -= mu_A.mean()
        mu_B -= mu_B.mean()

        diff = mu_A - mu_B
        avg  = 0.5 * (mu_A + mu_B)
        dd = 1
        vE = np.var(diff, ddof=dd)                 # noise variance (split-diff proxy)
        vU = np.var(avg,  ddof=dd)                 # signal variance (split-avg proxy)
        r_nc = (np.sqrt(vU) / np.sqrt(vU + vE)) if (vU + vE) > 1e-20 else np.nan

    # Normalize trialwise r by scalar ceiling
    trial_r_norm = trial_r / r_nc
    #trial_r_norm = np.where(np.isfinite(trial_r_norm), np.clip(trial_r_norm, -1.0, 1.0), np.nan)

    return {
        "source_roi": int(SOURCE_ROI),
        "target_roi": int(target_roi_id),
        "n_source_vox": int(X_conn.shape[1]),
        "n_target_vox": int(Y_conn.shape[1]),
        "trialwise_r": trial_r,                 # raw per-trial Pearson r
        "trialwise_r_norm": trial_r_norm,       # normalized by split-half noise ceiling
        "r_nc": float(r_nc) if np.isfinite(r_nc) else np.nan,
        "mean_r": float(np.nanmean(trial_r)),
        "mean_r_norm": float(np.nanmean(trial_r_norm)),
    }

combined_det_rows = []  
# two split settings: keep your tested default, plus the reverse CV of second half predicting first half
split_settings = [(SPLIT_CONN, SPLIT_PRED), ("first", "second")]
# de-duplicate in case defaults already equal ("first","second")
seen = set(); split_settings = [s for s in split_settings if (s not in seen and not seen.add(s))]

for split_conn_cur, split_pred_cur in split_settings:
    print("\n" + "#"*80)
    print(f"# Running split setting: CONN={split_conn_cur!r}, PRED={split_pred_cur!r}")
    print("#"*80)

    # Compute half-slices for this split (kept identical to your code)
    T = TS_PROXY.shape[-1]
    mid_T = T // 2
    if split_conn_cur == "first":
        CONN_SLICE = slice(0, mid_T)
    elif split_conn_cur == "second":
        CONN_SLICE = slice(mid_T, T)
    else:
        raise ValueError("SPLIT_CONN must be 'first' or 'second'.")

    N = BETA_PROXY.shape[-1]
    mid_N = N // 2
    if split_pred_cur == "first":
        PRED_SLICE = slice(0, mid_N)
    elif split_pred_cur == "second":
        PRED_SLICE = slice(mid_N, N)
    else:
        raise ValueError("SPLIT_PRED must be 'first' or 'second'.")

    print(f" CONN half: {split_conn_cur} -> {CONN_SLICE}")
    print(f" PRED half: {split_pred_cur} -> {PRED_SLICE}")

    # Loop over requested source ROIs (kept structure)
    for SOURCE_ROI in SOURCE_ROI_IDS:
        print(f"Source ROI: {SOURCE_ROI}")
        # derive targets for THIS source (exclude the source). list of target ROIs change per source.
        if TARGET_ROI_IDS is None:
            TARGETS_THIS = [i for i in ALL_ROI_IDS if i != int(SOURCE_ROI)]
        else:
            TARGETS_THIS = [int(i) for i in TARGET_ROI_IDS if int(i) != int(SOURCE_ROI)]

        print(" Running over predictions...")
        # run parallel for source predicting targets
        results = Parallel(n_jobs=int(N_JOBS), backend="loky", verbose=10)( delayed(predict_target_roi)(int(tid)) for tid in TARGETS_THIS )
        print(f" Completed {len(results)} targets for source {SOURCE_ROI}.")

        for r in results:
            tid = int(r["target_roi"])
            tw  = np.asarray(r["trialwise_r"]).ravel()
            twn = np.asarray(r["trialwise_r_norm"]).ravel()
            rl  = r.get("r_nc", np.nan)
            for t_idx, (rv, rn) in enumerate(zip(tw, twn)):
                if split_pred_cur == "second":
                    # if predicting second half, then trial index needs to be shifted by mid_N
                    t_idx += mid_N
                combined_det_rows.append({
                    "subject":      subject,
                    "split_conn":   split_conn_cur,
                    "split_pred":   split_pred_cur,
                    "source_roi":   int(SOURCE_ROI),
                    "target_roi":   tid,
                    "trial_index":  t_idx,
                    "r":            float(rv),
                    "r_norm":       float(rn),
                    "r_nc":         float(rl),
                    "n_source_vox": int(r["n_source_vox"]),
                    "n_target_vox": int(r["n_target_vox"]),
                })

# ===== save the single combined CSV for the full sample =====
combined_df = pd.DataFrame(combined_det_rows)
combined_out = OUTPUT_DIR + f"/{subject}_actflow_r_ALL.csv"
combined_df.to_csv(combined_out, index=False)
print("Done")
# end
