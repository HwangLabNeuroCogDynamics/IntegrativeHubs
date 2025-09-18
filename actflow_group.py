#!/usr/bin/env python3
"""
Group stats for activity flow
"""
import os
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from joblib import Parallel, delayed


ACTFLOW_DIR = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/actflow"
BEHAV_PATH = "/Shared/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_zRTs_full2_sorted_trialidx.csv"
ROI_MASK_PATH = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/ROIs/Morel_Striatum_Yeo400.nii.gz"
OUT_DIR = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/actflow_group"
BEH_SUBJ_COL = "sub"
BEH_TRIAL_COL = "Trial_idx"
BEH_TYPE_COL = "trial_type_v2"  #this has the same and diff task Steph wanted

# Parallel processing
N_JOBS = 24
# Trial conditions (IDS is reference)
TRIAL_CONDITIONS = ['IDS', 'Stay', 'Repeat', 'EDS_SameTask', 'EDS_DiffTask']
TEST_CONDITIONS = ['Stay', 'Repeat', 'EDS_SameTask', 'EDS_DiffTask']  

# =========================
# FUNCTIONS
# =========================
def fit_subject_level_model(subj_data, subject_id):
    """
    Fit OLS within a single subject using R-style formula
    Now handles: IDS (ref), Stay, Repeat, EDS_SameTask, EDS_DiffTask
    """
    trial_counts = subj_data['Trial_Type'].value_counts()    
    try:
        subj_data = subj_data.copy()
        subj_data['Trial_Type'] = pd.Categorical(
            subj_data['Trial_Type'], 
            categories=TRIAL_CONDITIONS, 
            ordered=False
        )
        
        fit = smf.ols('r_norm ~ 1 + C(Trial_Type, Treatment(reference="IDS"))', data=subj_data).fit()
        
        result = {
            'subject': subject_id,
            'n_trials': len(subj_data),
            'n_conditions': len(trial_counts),
            'trials_per_condition': dict(trial_counts),
            'r_squared': float(fit.rsquared),
            'converged': True
        }
        
        params = fit.params
        
        # Extract all condition effects vs IDS
        for condition in TEST_CONDITIONS:
            param_name = f'C(Trial_Type, Treatment(reference="IDS"))[T.{condition}]'
            if param_name in params.index:
                result[f'beta_{condition}'] = float(params[param_name])
                result[f'se_{condition}'] = float(fit.bse[param_name])
            else:
                result[f'beta_{condition}'] = np.nan
                result[f'se_{condition}'] = np.nan
        
        result['beta_intercept'] = float(params['Intercept'])
        
        return result
        
    except Exception as e:
        return {
            'subject': subject_id,
            'n_trials': len(subj_data),
            **{f'beta_{cond}': np.nan for cond in TEST_CONDITIONS},
            **{f'se_{cond}': np.nan for cond in TEST_CONDITIONS},
            'beta_intercept': np.nan,
            'r_squared': np.nan,
            'converged': False,
            'error': str(e)
        }

def apply_fdr_correction(stats_df):
    """
    Apply FDR correction to p-values for all test conditions
    """
    stats_df = stats_df.copy()
    
    # FDR correction for each condition
    for condition in TEST_CONDITIONS:
        p_col = f'p_{condition}'
        q_col = f'q_{condition}'
        sig_col = f'fdr_sig_{condition}'
        
        condition_pvals = stats_df[p_col].dropna()
        if len(condition_pvals) > 0:
            rejected, qvals = fdrcorrection(condition_pvals, alpha=0.05, method='indep')
            
            stats_df[q_col] = np.nan
            stats_df[sig_col] = False
            
            valid_idx = stats_df[p_col].notna()
            stats_df.loc[valid_idx, q_col] = qvals
            stats_df.loc[valid_idx, sig_col] = rejected
        else:
            stats_df[q_col] = np.nan
            stats_df[sig_col] = False
    
    return stats_df

def fit_two_level_model(roi_data, target_roi, source_roi):
    """
    Two-level analysis:
    1. Fit OLS within each subject
    2. Test subject-wise betas against 0 at group level
    """
    
    subjects = roi_data['subject_norm'].unique()
    
    try:
        # Level 1: Fit models within each subject
        subject_betas = []
        
        for subject in subjects:
            subj_data = roi_data[roi_data['subject_norm'] == subject].copy()
            subj_result = fit_subject_level_model(subj_data, subject)
            
            if subj_result is not None and subj_result['converged']:
                subject_betas.append(subj_result)
                
        beta_df = pd.DataFrame(subject_betas)
        
        # Level 2: Group-level tests (subject betas against 0)
        result = {
            "source_roi": int(source_roi),
            "target_roi": int(target_roi),
            "n_subjects_total": len(subjects),
            "n_subjects_converged": len(beta_df)
        }
        
        # Test each condition against 0
        for condition in TEST_CONDITIONS:
            condition_betas = beta_df[f'beta_{condition}'].dropna()
            t_stat, p_val = stats.ttest_1samp(condition_betas, 0)
            result.update({
                f"mean_beta_{condition}": float(condition_betas.mean()),
                f"se_beta_{condition}": float(condition_betas.std() / np.sqrt(len(condition_betas))),
                f"t_{condition}": float(t_stat),
                f"p_{condition}": float(p_val),
                f"n_subjects_{condition}": len(condition_betas)
            })

        
        result["mean_r_squared"] = float(beta_df['r_squared'].mean())
        result["mean_trials_per_subject"] = float(beta_df['n_trials'].mean())
        result["converged"] = True
        
        # Store individual subject betas for potential additional analyses
        for condition in TEST_CONDITIONS:
            condition_betas = beta_df[f'beta_{condition}'].dropna()
            result[f"individual_betas_{condition}"] = condition_betas.tolist()
        
        return result
        
    except Exception as e:
        error_result = {
            "source_roi": int(source_roi),
            "target_roi": int(target_roi),
            "n_subjects_total": len(roi_data['subject_norm'].unique()) if 'subject_norm' in roi_data.columns else 0,
            "converged": False,
            "error": str(e)
        }
        
        # Add NaN entries for all conditions
        for condition in TEST_CONDITIONS:
            error_result.update({
                f"mean_beta_{condition}": np.nan,
                f"se_beta_{condition}": np.nan,
                f"t_{condition}": np.nan,
                f"p_{condition}": np.nan
            })
        
        return error_result

def write_nii_from_roi_stats(stats_df, roi_mask_path, out_dir, source_roi):
    """
    Create NIfTI brain maps from ROI statistics for all test conditions
    """
    atlas_img = nib.load(roi_mask_path)
    atlas = atlas_img.get_fdata().astype(np.int32)
    
    n_filled = 0
    condition_fdr_counts = {}
    
    for condition in TEST_CONDITIONS:
        # Initialize maps
        t_map = np.zeros_like(atlas, dtype=np.float32)
        p_map = np.ones_like(atlas, dtype=np.float32)
        beta_map = np.zeros_like(atlas, dtype=np.float32)
        
        # Initialize FDR maps
        t_map_fdr = np.zeros_like(atlas, dtype=np.float32)
        q_map = np.ones_like(atlas, dtype=np.float32)
        beta_map_fdr = np.zeros_like(atlas, dtype=np.float32)
        
        n_fdr_condition = 0
        
        for _, row in stats_df.iterrows():
            roi = int(row["target_roi"])
            mask = (atlas == roi)
            
            if np.any(mask):
                # Uncorrected maps
                if np.isfinite(row[f"t_{condition}"]):
                    t_map[mask] = row[f"t_{condition}"]
                    beta_map[mask] = row[f"mean_beta_{condition}"]
                if np.isfinite(row[f"p_{condition}"]):
                    p_map[mask] = row[f"p_{condition}"]
                
                # FDR-corrected maps
                if row.get(f"fdr_sig_{condition}", False):
                    t_map_fdr[mask] = row[f"t_{condition}"]
                    beta_map_fdr[mask] = row[f"mean_beta_{condition}"]
                    n_fdr_condition += 1
                if np.isfinite(row.get(f"q_{condition}", np.nan)):
                    q_map[mask] = row[f"q_{condition}"]
        
        condition_fdr_counts[condition] = n_fdr_condition
        
        # Save images
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        # Uncorrected maps
        nib.save(nib.Nifti1Image(t_map, atlas_img.affine, atlas_img.header),
                 os.path.join(out_dir, f"t_{condition}_vs_IDS_src{source_roi}.nii.gz"))
        nib.save(nib.Nifti1Image(p_map, atlas_img.affine, atlas_img.header),
                 os.path.join(out_dir, f"p_{condition}_vs_IDS_src{source_roi}.nii.gz"))
        nib.save(nib.Nifti1Image(beta_map, atlas_img.affine, atlas_img.header),
                 os.path.join(out_dir, f"beta_{condition}_vs_IDS_src{source_roi}.nii.gz"))
        
        # FDR-corrected maps
        nib.save(nib.Nifti1Image(t_map_fdr, atlas_img.affine, atlas_img.header),
                 os.path.join(out_dir, f"t_{condition}_vs_IDS_FDR_src{source_roi}.nii.gz"))
        nib.save(nib.Nifti1Image(beta_map_fdr, atlas_img.affine, atlas_img.header),
                 os.path.join(out_dir, f"beta_{condition}_vs_IDS_FDR_src{source_roi}.nii.gz"))
        nib.save(nib.Nifti1Image(q_map, atlas_img.affine, atlas_img.header),
                 os.path.join(out_dir, f"q_{condition}_vs_IDS_src{source_roi}.nii.gz"))
    
    n_filled = len(stats_df)
    return n_filled, condition_fdr_counts

# =========================
# MAIN SCRIPT
# =========================
if __name__ == "__main__":
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Loading ActFlow CSVs...")
    actflow_files = sorted(glob(os.path.join(ACTFLOW_DIR, "*_actflow_r_ALL.csv")))
    
    all_actflow = []
    for fp in actflow_files:
        df = pd.read_csv(fp)
        df["subject_norm"] = df["subject"].astype(str).str.replace('sub-', '', regex=False)
        # Simplified global trial index (assuming consistent indexing)
        df["global_trial"] = df["trial_index"]
        all_actflow.append(df)
    
    actflow_data = pd.concat(all_actflow, ignore_index=True)
    print(f"   Loaded {len(actflow_data)} rows from {actflow_data['subject'].nunique()} subjects")
    
    print("Loading behavior CSV...")
    behavior_data = pd.read_csv(BEHAV_PATH)
    
    # Normalize subject ID and trial index
    behavior_data["subject_norm"] = behavior_data[BEH_SUBJ_COL].astype(str).str.replace('sub-', '', regex=False)
    behavior_data["global_trial"] = behavior_data[BEH_TRIAL_COL]
    print("[3/6] Merging ActFlow with behavior...")
    merged_data = actflow_data.merge(
        behavior_data,
        on=["subject_norm", "global_trial"],
        how="inner"
    )    
    print(f"   Before filtering: {len(merged_data)} rows")
    
    print("Filtering data...")
    
    # 1. Exclude slow responses (zRT > 3)
    if 'zRT' in merged_data.columns:
        merged_data = merged_data[merged_data['zRT'] <= 3].copy()
        print(f"   After zRT filter: {len(merged_data)} rows")
    else:
        print("   Warning: 'zRT' column not found - skipping RT filter")
    
    # 2. Exclude incorrect responses
    if 'trial_Corr' in merged_data.columns:
        merged_data = merged_data[merged_data['trial_Corr'] == 1].copy()
        print(f"   After accuracy filter: {len(merged_data)} rows")
    else:
        print("   Warning: 'trial_Corr' column not found - skipping accuracy filter")
        
    # Use trial_type_v2 and exclude NaN
    merged_data = merged_data[merged_data[BEH_TYPE_COL].notna()].copy()
    print(f"   After removing NaN conditions: {len(merged_data)} rows")
    
    merged_data['Trial_Type'] = pd.Categorical( merged_data[BEH_TYPE_COL], categories=TRIAL_CONDITIONS, ordered=False )
    print(f"   Final filtered data: {len(merged_data)} rows")
    print(f"   Trial type counts:\n{merged_data['Trial_Type'].value_counts()}")
        
    print(f"   Final merged data: {len(merged_data)} rows (subjects: {merged_data['subject_norm'].nunique()}, "
          f"targets: {merged_data['target_roi'].nunique()})")
    
    print("Running two-level models per source ROI...")
    source_rois = sorted(merged_data["source_roi"].unique())
    print(f"   Found {len(source_rois)} source ROIs: {source_rois}")
    
    all_results = []
    
    for source_roi in source_rois:
        print(f"\n   Processing source ROI {source_roi}...")
        
        source_data = merged_data[merged_data["source_roi"] == source_roi].copy()
        
        if len(source_data) == 0:
            print(f"     Warning: No data for source ROI {source_roi}")
            continue
        
        target_rois = sorted(source_data["target_roi"].unique())
        print(f"     Processing {len(target_rois)} target ROIs...")
        
        roi_results = Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(fit_two_level_model)(
                source_data[source_data["target_roi"] == target_roi].copy(),
                target_roi,
                source_roi
            )
            for target_roi in target_rois
        )
        
        roi_results = [r for r in roi_results if r is not None]
        
        if roi_results:
            stats_df = pd.DataFrame(roi_results)
            
            print(f"     Applying FDR correction...")
            stats_df = apply_fdr_correction(stats_df)
            
            all_results.append(stats_df)
            
            stats_path = os.path.join(OUT_DIR, f"group_stats_two_level_src{source_roi}.csv")
            stats_df.to_csv(stats_path, index=False)
            print(f"     Saved: {stats_path}")
            
            n_filled, condition_fdr_counts = write_nii_from_roi_stats(stats_df, ROI_MASK_PATH, OUT_DIR, source_roi)
            print(f"     Created brain maps with {n_filled} ROIs filled")
            
            # Print FDR summary for each condition
            for condition in TEST_CONDITIONS:
                fdr_count = condition_fdr_counts.get(condition, 0)
                uncorr_count = (stats_df[f"p_{condition}"] < 0.05).sum()
                print(f"     {condition}: {uncorr_count} uncorrected, {fdr_count} FDR-corrected")
            
            converged = stats_df["converged"].sum()
            total = len(stats_df)
            print(f"     Summary: {converged}/{total} models converged")
        else:
            print(f"     No valid models for source ROI {source_roi}")
        
    print("\nDone!")