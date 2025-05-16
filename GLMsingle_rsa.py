import numpy as np
import pandas as pd
import os
import nibabel as nib
from datetime import datetime
import scipy
import scipy.linalg # Explicitly import scipy.linalg
# from nilearn.maskers import NiftiLabelsMasker # Not strictly needed if mask is single region or handled differently
from nilearn import image as nilearn_image # For new_img_like
from nilearn import masking as nilearn_masking # For apply_mask
from scipy.spatial.distance import pdist, squareform


################################
## Calculate trial level RSA.
################################
included_subjects = input("Enter subject ID (e.g., 10001): ")


def compute_shrunk_covariance_and_whitening_matrix(x, return_shrunk_cov=False):
    """
    Computes the shrunk covariance matrix and its inverse square root for whitening.
    Optionally returns the shrunk covariance matrix itself (for Mahalanobis).

    Args:
        x (np.array): Data matrix, rows are observations (e.g., beta deviation patterns),
                      columns are features (e.g., voxels). Shape (t, n).
        return_shrunk_cov (bool): If True, also returns the shrunk covariance matrix.

    Returns:
        inv_sigma_sqrt (np.array): Whitening matrix (Sigma_shrunk)^(-1/2). Shape (n, n).
        sigma_shrunk (np.array, optional): Shrunk covariance matrix. Shape (n, n).
                                           Returned if return_shrunk_cov is True.
    """
    t, n = x.shape # t measurements by n voxels
    if t < 2:
        print(f"Warning: Not enough samples ({t}) to compute covariance reliably (need at least 2). Returning identity-based matrices.")
        identity_matrix = np.eye(n)
        if return_shrunk_cov:
            return identity_matrix, identity_matrix 
        else:
            return identity_matrix

    x_demeaned = x - x.mean(0) # Ensure data is demeaned for covariance calculation
    sample_cov = (1.0 / (t - 1)) * np.dot(np.transpose(x_demeaned), x_demeaned)
    prior = np.diag(np.diag(sample_cov))
    
    # Simplified shrinkage - this part can be critical and might need tuning/more advanced methods
    # Using a more stable shrinkage approach based on Ledoit-Wolf intuition without direct sklearn dependency for this snippet
    # Calculate sum of squared errors for sample_cov relative to prior
    num = np.sum((sample_cov - prior)**2)
    # Calculate sum of squared errors for each observation relative to its mean (related to variance of variances)
    # This is a simplified denominator
    den = np.sum((x_demeaned**2 - np.diag(sample_cov)[np.newaxis,:])**2) / t
    
    shrinkage = 1.0 # Default to full shrinkage (prior) if denominator is zero or num is zero
    if den > 1e-10 and num > 1e-10 : # Avoid division by zero
        shrinkage = num / den
    shrinkage = np.clip(shrinkage, 0.0, 1.0)

    # Heuristic to increase shrinkage if t is small relative to n (more voxels than samples)
    if t < n:
        print(f"Warning: Number of deviation samples ({t}) is less than number of voxels ({n}). Increasing shrinkage.")
        shrinkage = max(shrinkage, 1 - ( (t-1) / (n*1.5) ) ) # Ensure t-1 > 0
        shrinkage = np.clip(shrinkage, 0.0, 1.0)


    sigma_shrunk = shrinkage * prior + (1 - shrinkage) * sample_cov
    
    epsilon = 1e-7 # Small constant to add to diagonal for numerical stability
    try:
        # Check for positive definiteness before inversion attempt
        np.linalg.cholesky(sigma_shrunk)
    except np.linalg.LinAlgError:
        print("Warning: Shrunk covariance not positive definite. Adding epsilon to diagonal.")
        sigma_shrunk_reg = sigma_shrunk + epsilon * np.eye(n)
        # Try again with regularized version
        try:
          np.linalg.cholesky(sigma_shrunk_reg)
          sigma_shrunk = sigma_shrunk_reg # Use regularized if it works
        except np.linalg.LinAlgError:
          print("Warning: Regularized shrunk covariance also not positive definite. Using prior for whitening.")
          sigma_shrunk = prior + epsilon * np.eye(n) # Fallback to regularized prior

    whitening_matrix = np.eye(n) # Initialize to identity as fallback
    try:
        evals, evecs = np.linalg.eigh(sigma_shrunk)
        # Floor eigenvalues to prevent issues with very small or negative ones from numerical precision
        evals_floored = np.maximum(evals, epsilon) 
        evals_sqrt_inv = 1.0 / np.sqrt(evals_floored)
        whitening_matrix = np.dot(evecs * evals_sqrt_inv, evecs.T)
    except np.linalg.LinAlgError as err:
        print(f"Error during eigendecomposition/inversion of shrunk covariance: {err}")
        print("Falling back to using whitening matrix based on prior (diagonal matrix).")
        try:
            evals_prior, evecs_prior = np.linalg.eigh(prior) # prior is diagonal
            evals_prior_floored = np.maximum(evals_prior, epsilon)
            evals_prior_sqrt_inv = 1.0 / np.sqrt(evals_prior_floored)
            whitening_matrix = np.dot(evecs_prior * evals_prior_sqrt_inv, evecs_prior.T)
            if return_shrunk_cov:
                sigma_shrunk = prior # If whitening failed, Mahalanobis on shrunk_cov might also be unstable
        except np.linalg.LinAlgError:
            print("Fallback to prior also failed. Using identity for whitening matrix.")
            whitening_matrix = np.eye(n)
            if return_shrunk_cov:
                sigma_shrunk = np.eye(n)

    if return_shrunk_cov:
        return whitening_matrix, sigma_shrunk
    else:
        return whitening_matrix


if __name__ == "__main__":

    now = datetime.now()
    print("Start time: ", now)
    
    #### setup
    mask_dir = "/Shared/lss_kahwang_hpc/ROIs/"
    GLMsingle_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/"
    out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/trialwiseRSA/"
    
    roi_fn_out_suffix = "Schaefer400_Morel_BG" 
    ROI_mask_name = "Schaefer400+Morel+BG_2.5.nii.gz" 

    mask_file = os.path.join(mask_dir, ROI_mask_name)
    print("\n\nLoading mask file from ... ", mask_file)
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() # This is the label data from the atlas
    print("Mask shape ", mask_data.shape)
    
    #os.makedirs(os.path.join(out_dir, "rdms_correlation_whitened_betas"), exist_ok=True)
    #os.makedirs(os.path.join(out_dir, "rdms_euclidean_whitened_betas"), exist_ok=True)
    #os.makedirs(os.path.join(out_dir, "rdms_mahalanobis_betas"), exist_ok=True)

    for s_idx, s in enumerate([included_subjects]):
        
        print(f"\n--- Processing Subject: {s} ---")

        # Load GLMsingle trial betas
        print(f"\nLoading GLMsingle TrialBetas for subject {s}")
        trial_betas_nii_path = os.path.join(GLMsingle_dir, f"sub-{s}", f"{s}_TrialBetas_resRT.nii.gz")
        if not os.path.exists(trial_betas_nii_path):
            print(f"TrialBetas file not found for subject {s} at {trial_betas_nii_path}. Skipping.")
            continue
        trial_betas_nii = nib.load(trial_betas_nii_path)
        num_trials = trial_betas_nii.shape[3]
        print(f"Number of trials found in beta file: {num_trials}")
            
        # Create a binary mask from the atlas data (all labeled regions become part of the mask)
        overall_binary_mask_data = np.where(mask_data > 0, 1.0, 0.0)
        print(f"Total number of voxels in initial ROI atlas (excluding background): {overall_binary_mask_data.sum()}\n")
        if overall_binary_mask_data.sum() == 0:
            print("No voxels found in the provided ROI atlas. Skipping subject.")
            continue
        overall_binary_mask_img = nilearn_image.new_img_like(mask_img, overall_binary_mask_data)

        # Mask the GLMsingle trial betas to get [trials x all_voxels_in_atlas]
        lss_masked_trials_x_voxels = nilearn_masking.apply_mask(trial_betas_nii, overall_binary_mask_img)
        # Transpose to [all_voxels_in_atlas x trials] for easier processing
        all_voxels_betas_V_x_T = lss_masked_trials_x_voxels.T
        num_total_voxels_in_atlas = all_voxels_betas_V_x_T.shape[0]
        
        print(f"Shape of masked trial betas (all atlas voxels x trials): {all_voxels_betas_V_x_T.shape}")

        # --- Prepare beta deviations for covariance estimation ---
        print("Calculating beta deviations ...")
        # Mean beta pattern across ALL trials: [V x 1]
        mean_all_trials_beta_V_x_1 = np.mean(all_voxels_betas_V_x_T, axis=1, keepdims=True)
        # Deviation patterns: [V x N_total_trials]
        all_beta_deviations_V_x_T = all_voxels_betas_V_x_T - mean_all_trials_beta_V_x_1
        # For compute_shrunk_covariance, we need [Samples x Voxels], so [N_total_trials x V]
        beta_deviations_for_cov_T_x_V = all_beta_deviations_V_x_T.T
        print(f"Shape of stacked beta deviations for covariance (trials x all atlas voxels): {beta_deviations_for_cov_T_x_V.shape}")

        # Get unique ROI labels present in the atlas mask_data (within the binary mask)
        # These are the actual integer labels from Schaefer400+Morel+BG_2.5.nii.gz
        # voxel_roi_labels_for_masked_data will be a 1D array mapping each voxel in all_voxels_betas_V_x_T to its ROI ID
        voxel_roi_labels_for_masked_data = nilearn_masking.apply_mask(mask_img, overall_binary_mask_img)
        
        unique_roi_ids = np.unique(voxel_roi_labels_for_masked_data)
        unique_roi_ids = unique_roi_ids[unique_roi_ids > 0] # Exclude background label 0
        num_ROIs = len(unique_roi_ids)
        print(f"Found {num_ROIs} unique ROI labels in the atlas: {unique_roi_ids}")
        if num_ROIs == 0:
            print("No valid ROI labels found after masking. Skipping subject.")
            continue

        # Initialize RDM storage
        rdm_corr_whitened = np.full((num_ROIs, num_trials, num_trials), np.nan)
        rdm_euc_whitened = np.full((num_ROIs, num_trials, num_trials), np.nan)
        #rdm_mahalanobis = np.full((num_ROIs, num_trials, num_trials), np.nan)

        for i, current_roi_id in enumerate(unique_roi_ids):
            print(f"\n--- Working on ROI ID: {current_roi_id} ({i+1}/{num_ROIs}) ---")
            
            # Indices of voxels belonging to the current ROI within the already masked data
            cur_roi_voxel_indices = np.where(voxel_roi_labels_for_masked_data == current_roi_id)[0]
            num_voxels_in_roi = len(cur_roi_voxel_indices)
            print(f"Number of voxels in current ROI {current_roi_id}: {num_voxels_in_roi}")

            if num_voxels_in_roi == 0: # Should not happen if unique_roi_ids is derived correctly
                print("ROI has no voxels. Skipping.")
                continue
            if num_voxels_in_roi == 1 and num_trials > 1:
                 print("ROI has only 1 voxel. Whitening/Mahalanobis not applicable in multivariate sense. Correlation RDM will be all zeros (or NaN). Euclidean RDM will be computed.")
                 # Fallback for single voxel: No multivariate structure to whiten/account for
                 roi_betas_1voxel_x_T = all_voxels_betas_V_x_T[cur_roi_voxel_indices, :] # Shape (1, num_trials)
                 try:
                     rdm_euc_whitened[i, :, :] = squareform(pdist(roi_betas_1voxel_x_T.T, metric='euclidean'))
                 except Exception as e:
                     print(f"Error computing Euclidean for 1-voxel ROI: {e}")
                 # Correlation for 1 feature doesn't make sense for an RDM of trial dissimilarities based on patterns.
                 # Mahalanobis also doesn't make sense in a typical way.
                 # We'll leave them as NaN which is the default value of the rdm arrays.
                 continue # Skip to next ROI

            # Select beta deviations for the current ROI: [Trials x ROI_voxels]. This is for noise cov.
            beta_devs_roi_T_x_Vroi = beta_deviations_for_cov_T_x_V[:, cur_roi_voxel_indices]
            
            if beta_devs_roi_T_x_Vroi.shape[0] < 2 : # samples (trials) should be at least 2
                print(f"Not enough trials ({beta_devs_roi_T_x_Vroi.shape[0]}) for ROI {current_roi_id} to estimate covariance. Using identity for whitening/covariance.")
                whitening_matrix_roi = np.eye(num_voxels_in_roi)
                shrunk_cov_roi = np.eye(num_voxels_in_roi) 
            else:
                whitening_matrix_roi, shrunk_cov_roi = compute_shrunk_covariance_and_whitening_matrix( beta_devs_roi_T_x_Vroi, return_shrunk_cov=True )
            # print(f"Whitening matrix shape for ROI {current_roi_id}: {whitening_matrix_roi.shape}")
            # print(f"Shrunk covariance matrix shape for ROI {current_roi_id}: {shrunk_cov_roi.shape}")

            # Get original beta patterns for the current ROI: [ROI_voxels x trials]
            roi_betas_Vroi_x_T = all_voxels_betas_V_x_T[cur_roi_voxel_indices, :]
            # print(f"Shape of betas for ROI {current_roi_id} (ROI_voxels x trials): {roi_betas_Vroi_x_T.shape}")

            # Whiten the betas for this ROI: [ROI_voxels_whitened x trials]
            whitened_roi_betas_Vroi_x_T = np.dot(whitening_matrix_roi, roi_betas_Vroi_x_T)
            # print(f"Shape of whitened betas for ROI {current_roi_id}: {whitened_roi_betas_Vroi_x_T.shape}")

            # --- Calculate RDMs ---
            # Data for pdist should be [samples x features] = [trials x voxels]
            data_for_pdist_whitened = whitened_roi_betas_Vroi_x_T.T
            data_for_pdist_original = roi_betas_Vroi_x_T.T

            # 1. Correlation on whitened betas
            if num_voxels_in_roi > 1 : # Correlation needs at least 2 features (voxels) to be meaningful for pattern similarity
                try:
                    # For np.corrcoef, if input is M_voxels x N_trials, use rowvar=False for N_trials x N_trials RDM
                    corr_values = np.corrcoef(whitened_roi_betas_Vroi_x_T, rowvar=False)
                    if np.any(np.isnan(corr_values)): # Check if corrcoef itself produced NaNs (e.g. zero variance after whitening)
                        print(f"Warning: NaNs in correlation matrix for ROI {current_roi_id} after whitening.")
                        rdm_corr_whitened[i, :, :] = np.nan
                    else:
                        rdm_corr_whitened[i, :, :] = corr_values
                except Exception as e_corr:
                    print(f"Error computing correlation RDM for ROI {current_roi_id}: {e_corr}")
                    rdm_corr_whitened[i, :, :] = np.nan
            else: # if only one voxel, correlation distance isn't well-defined for pattern similarity
                 rdm_corr_whitened[i, :, :] = np.nan


            # 2. Euclidean distance on whitened betas
            try:
                rdm_euc_whitened[i, :, :] = squareform(pdist(data_for_pdist_whitened, metric='euclidean'))
            except Exception as e_euc:
                print(f"Error computing Euclidean RDM for ROI {current_roi_id}: {e_euc}")
                rdm_euc_whitened[i, :, :] = np.nan
            
            ## below is the same as euclidean, so skipping.
            # 3. Mahalanobis distance on ORIGINAL (non-whitened) betas using INVERSE of shrunk_cov_roi. 
            # if num_voxels_in_roi > 0: # Mahalanobis needs at least one feature
            #     try:
            #         # pdist requires the INVERSE of the covariance matrix
            #         inv_shrunk_cov_roi = np.linalg.inv(shrunk_cov_roi)
            #         rdm_mahalanobis[i, :, :] = squareform(pdist(data_for_pdist_original,
            #                                                     metric='mahalanobis',
            #                                                     VI=inv_shrunk_cov_roi))
            #     except np.linalg.LinAlgError:
            #         print(f"Could not invert shrunk_cov_roi for Mahalanobis in ROI {current_roi_id}. Using pseudo-inverse or fallback.")
            #         try:
            #             # Try pseudo-inverse as a fallback
            #             pinv_shrunk_cov_roi = np.linalg.pinv(shrunk_cov_roi)
            #             rdm_mahalanobis[i, :, :] = squareform(pdist(data_for_pdist_original,
            #                                                         metric='mahalanobis',
            #                                                         VI=pinv_shrunk_cov_roi))
            #             print("Used pseudo-inverse for Mahalanobis.")
            #         except Exception as e_pinv:
            #              print(f"Pseudo-inverse also failed for Mahalanobis: {e_pinv}. RDM for Mahalanobis will be NaN.")
            #              rdm_mahalanobis[i, :, :] = np.nan # Ensure it's NaN
            #     except ValueError as ve: 
            #         print(f"ValueError during Mahalanobis for ROI {current_roi_id} (e.g. zero variance): {ve}. RDM will be NaN.")
            #         rdm_mahalanobis[i, :, :] = np.nan
            #     except Exception as e_mah: # Catch any other unexpected errors
            #         print(f"An unexpected error occurred during Mahalanobis for ROI {current_roi_id}: {e_mah}")
            #         rdm_mahalanobis[i, :, :] = np.nan
            # else: # Should be caught by num_voxels_in_roi == 0 earlier
            #     rdm_mahalanobis[i, :, :] = np.nan


        # Save RDMs for the current subject
        if num_ROIs > 0 : # Only save if there was at least one ROI processed
            np.save(os.path.join(out_dir, "rdms_correlation_whitened_betas", f"{s}_{roi_fn_out_suffix}_rdm_corr_whitened_resRT.npy"), rdm_corr_whitened)
            np.save(os.path.join(out_dir, "rdms_euclidean_whitened_betas", f"{s}_{roi_fn_out_suffix}_rdm_euc_whitened_resRT.npy"), rdm_euc_whitened)
            #np.save(os.path.join(out_dir, "rdms_mahalanobis_betas", f"{s}_{roi_fn_out_suffix}_rdm_mahalanobis.npy"), rdm_mahalanobis)
            print(f"RDMs saved for subject {s}")

    now_end = datetime.now()
    print("\nEnd time: ", now_end)
    print("Total duration: ", now_end - now)

# end