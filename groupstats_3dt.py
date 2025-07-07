# run one-sample t-tests (3dttest++) from python
import os
import glob
import subprocess

#### model regression
regressors = [
"main_effect_EDS_v_IDS_ProbeFeatureRTAccuModel_resRTWTT",
"main_effect_IDS_v_Stay_ProbeFeatureRTAccuModel_resRTWTT",
"main_effect_perceptual_change_c_ProbeFeatureRTAccuModel_resRTWTT",
"main_effect_prev_target_feature_match_ProbeFeatureRTAccuModel_resRTWTT",
"main_effect_response_repeat_ProbeFeatureRTAccuModel_resRTWTT",
"main_effect_task_repeat_ProbeFeatureRTAccuModel_resRTWTT",
"perceptual_change_for_EDS_ProbeFeatureRTAccuModel_resRTWTT",
"perceptual_change_for_IDS_ProbeFeatureRTAccuModel_resRTWTT",
"perceptual_change_for_Stay_ProbeFeatureRTAccuModel_resRTWTT",
"response_repeat_T_True_C_prev_target_feature_match_vs_same_target_feature_ProbeFeatureRTAccuModel_resRTWTT",
"interaction_response_repeat_x_task_repeat_ProbeFeatureRTAccuModel_resRTWTT",
"task_repeat_T_True_C_prev_target_feature_match_vs_same_target_feature_ProbeFeatureRTAccuModel_resRTWTT",
"Errors_ProbeFeatureRTAccuModel_resRTWTT",
"zRT_ProbeFeatureRTAccuModel_resRTWTT",
"probe_repeat_vs_switch_ProbeFeatureRTAccuModel_resRTWTT",
]
data_dir   = '/home/kahwang/argon/data/ThalHi/GLMsingle/searchlightRSA_regression'
mask_path  = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA/searchlight_mask.nii.gz'
group_dir  = os.path.join(data_dir, "group_stats")
os.makedirs(group_dir, exist_ok=True)

# Choose one of: "Clustsim", "ETAC", or None
cluster_option = "Clustsim"
#cluster_option = None
# regressors = [
# "main_effect_EDS_v_IDS_ProbeFeatureRTAccuModel_resRTWTT",
# ]

for reg in regressors:
    beta_files = sorted(glob.glob(f"{data_dir}/*/{reg}_beta.nii.gz"))
    if not beta_files:
        print(f"⚠️  No betas for {reg}, skipping")
        continue

    # Use just the basename as prefix
    rel_prefix = f"group_{reg}_ttest"
    resid_prefix = f"group_{reg}_resid"
    cmd = ["3dttest++",
           "-mask", mask_path,
           "-prefix", rel_prefix]
    if cluster_option == "Clustsim":
        cmd += ["-Clustsim"]
    elif cluster_option == "ETAC":
        cmd += ["-ETAC"]
    elif cluster_option is None:
        cmd += ["-ACF -resid", resid_prefix]
    cmd += ["-setA"] + beta_files

    print(f"🚀 Running in {group_dir}:", " ".join(cmd))
    # run with cwd=group_dir so -prefix is relative to here
    subprocess.run(cmd, check=True, cwd=group_dir)
    print(f"✓ Done: {os.path.join(group_dir, rel_prefix)}+tlrc\n")



#### distance regression
regressors = ["distance" ]

data_dir   = '/home/kahwang/argon/data/ThalHi/GLMsingle/'
mask_path  = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA/searchlight_mask.nii.gz'
group_dir  = os.path.join(data_dir, "group_stats")
os.makedirs(group_dir, exist_ok=True)

# Choose one of: "Clustsim", "ETAC", or None
cluster_option = "Clustsim"

for reg in regressors:
    beta_files = sorted(glob.glob(f"{data_dir}/*/*{reg}_beta.nii.gz"))
    if not beta_files:
        print(f"⚠️  No betas for {reg}, skipping")
        continue

    # Use just the basename as prefix
    rel_prefix = f"group_{reg}_ttest"

    cmd = ["3dttest++",
           "-mask", mask_path,
           "-prefix", rel_prefix]
    if cluster_option == "Clustsim":
        cmd += ["-Clustsim"]
    elif cluster_option == "ETAC":
        cmd += ["-ETAC"]
    cmd += ["-setA"] + beta_files

    print(f"🚀 Running in {group_dir}:", " ".join(cmd))
    # run with cwd=group_dir so -prefix is relative to here
    subprocess.run(cmd, check=True, cwd=group_dir)
    print(f"✓ Done: {os.path.join(group_dir, rel_prefix)}+tlrc\n")