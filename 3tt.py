# run one-sample tests (3dMEMA) from python
import os
import glob
import subprocess

#### model regression
regressors = [
    "main_effect_EDS_v_IDS_ProbeFeatureRTAccuModel_resRTWTT",
    "main_effect_IDS_v_Stay_ProbeFeatureRTAccuModel_resRTWTT",
    "main_effect_perceptual_change_c_ProbeFeatureRTAccuModel_resRTWTT",
    "main_effect_prev_target_feature_switch_ProbeFeatureRTAccuModel_resRTWTT",
    "main_effect_response_repeat_ProbeFeatureRTAccuModel_resRTWTT",
    "main_effect_task_repeat_ProbeFeatureRTAccuModel_resRTWTT",
    "perceptual_change_for_EDS_ProbeFeatureRTAccuModel_resRTWTT",
    "perceptual_change_for_IDS_ProbeFeatureRTAccuModel_resRTWTT",
    "perceptual_change_for_Stay_ProbeFeatureRTAccuModel_resRTWTT",
    "interaction_response_repeat_x_task_repeat_ProbeFeatureRTAccuModel_resRTWTT",
    "task_repeat_T_switch_C_prev_target_feature_switch_vs_repeat_ProbeFeatureRTAccuModel_resRTWTT",
    "response_repeat_T_switch_C_prev_target_feature_switch_vs_repeat_ProbeFeatureRTAccuModel_resRTWTT",
    "main_effect_Errors_ProbeFeatureRTAccuModel_resRTWTT",
    "Prev_Errors_vs_error_ProbeFeatureRTAccuModel_resRTWTT",
    "zRT_ProbeFeatureRTAccuModel_resRTWTT",
    "zRT_diff_ProbeFeatureRTAccuModel_resRTWTT",
    "probe_repeat_vs_switch_ProbeFeatureRTAccuModel_resRTWTT",
]

# -------------------------------
# paths
# -------------------------------
if os.path.exists("/Shared"):
    data_dir  = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightNeuralDistance_RTdiff"
    mask_path = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA/searchlight_mask.nii.gz"
    afni_cmd  = ["singularity", "run", "--cleanenv",
                 "/Shared/lss_kahwang_hpc/opt/afni/afni.sif", "3dMEMA"]
else:
    data_dir  = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightNeuralDistance_RTdiff"
    mask_path = "/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA/searchlight_mask.nii.gz"
    afni_cmd  = ["3dMEMA"]

group_dir = os.path.join(data_dir, "group_stats_mema")
os.makedirs(group_dir, exist_ok=True)

# Choose cluster option: "None" or "Clustsim"
cluster_option = "None"

# -------------------------------
# main loop
# -------------------------------
for reg in regressors:

    beta_files = sorted(glob.glob(f"{data_dir}/*/{reg}_beta.nii.gz"))
    tval_files = sorted(glob.glob(f"{data_dir}/*/{reg}_tval.nii.gz"))

    if not beta_files or not tval_files:
        print(f" Missing beta or tvals for {reg}, skipping")
        continue

    # sanity check: subjects match
    bet_subs = [os.path.basename(os.path.dirname(b)) for b in beta_files]
    t_subs   = [os.path.basename(os.path.dirname(t)) for t in tval_files]

    if bet_subs != t_subs:
        print(f" Beta and tval subject ordering mismatch for {reg}")
        continue

    prefix = f"group_{reg}_MEMA"

    cmd = afni_cmd + [
        "-mask", mask_path,
        "-prefix", prefix,
        "-jobs", "24",
        "-model_outliers",
        "-residual_Z",
        "-set", reg
    ]

    # Add subjects:   subj  beta.nii.gz  tval.nii.gz
    for subj, b, t in zip(bet_subs, beta_files, tval_files):
        cmd += [subj, b, t]

    if cluster_option == "Clustsim":
        cmd.append("-Clustsim")

    print(f"======================================================")
    print(f"🚀 Running MEMA for: {reg}")
    print(" ".join(cmd))
    print(f"======================================================")

    subprocess.run(cmd, check=True, cwd=group_dir)

    print(f"✓ Finished MEMA: {os.path.join(group_dir, prefix)}\n")
