"""
Fast searchlight adjacency RSA script using a custom _apply_mask_and_get_affinity.
Computes Euclidean and correlation distances between adjacent trials within each sphere.
"""
import os
import warnings
import numpy as np
import nibabel as nib
from datetime import datetime
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn import neighbors
from nilearn import image, masking
from nilearn._utils.niimg_conversions import (
    check_niimg_3d,
    safe_get_data,
)
from nilearn.image.resampling import coord_transform
from joblib import Parallel, delayed
import nilearn

# ------------------------------------------------------------------------
# Fast affinity function adapted from nilearn
# ------------------------------------------------------------------------
def new_apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap, mask_img=None):
    """
    Build a sparse sphere-to-voxel affinity matrix.

    seeds      : array-like of shape (n_seeds, 3) in world (mm) coordinates
    niimg      : Nifti1Image of 4D data (for affine) or None
    radius     : float, searchlight radius in mm
    allow_overlap: bool, if False, raises on overlapping spheres
    mask_img   : Nifti1Image or filename of 3D binary mask

    Returns:
      X         : data matrix from mask_img & niimg (unused here)
      A_sparse  : lil_matrix (n_seeds x n_voxels) boolean
    """
    seeds = np.asarray(seeds)
    if seeds.ndim == 1:
        seeds = seeds[np.newaxis, :]

    # Load or derive mask coordinates and optional data
    if niimg is None:
        mask, affine = masking.load_mask_img(mask_img)
        mask_coords = np.column_stack(np.where(mask))
        X = None
    else:
        affine = niimg.affine
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(
            mask_img,
            target_affine=affine,
            target_shape=niimg.shape[:3],
            interpolation="nearest",
            force_resample=False,
        )
        mask, _ = masking.load_mask_img(mask_img)
        mask_coords = np.column_stack(np.where(mask != 0))
        X = masking.apply_mask_fmri(niimg, mask_img)

    # Convert mask voxel indices to world space
    mask_coords_world = coord_transform(
        mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2], affine
    )
    mask_coords_world = np.column_stack(mask_coords_world)

    # Map seeds from world → voxel indices
    inv_affine = np.linalg.inv(affine)
    seeds_voxel = (
        np.column_stack([seeds, np.ones(len(seeds))]) @ inv_affine.T
    )[:, :3]
    nearest_voxels = np.round(seeds_voxel).astype(int)
    coord_dict = {tuple(c): i for i, c in enumerate(mask_coords)}
    nearests = [coord_dict.get(tuple(coord), None) for coord in nearest_voxels]

    # KDTree for radius queries
    tree = neighbors.KDTree(mask_coords_world)
    neighbors_list = tree.query_radius(seeds, r=radius)

    n_seeds = len(seeds)
    n_voxels = len(mask_coords)
    A_sparse = sparse.lil_matrix((n_seeds, n_voxels), dtype=bool)

    # Fill boolean matrix
    for i, (nbrs, nearest) in enumerate(zip(neighbors_list, nearests)):
        if nbrs.size > 0:
            A_sparse[i, nbrs] = True
        if nearest is not None:
            A_sparse[i, nearest] = True

    # Ensure seed itself is present
    seed_world = coord_transform(
        seeds_voxel[:, 0], seeds_voxel[:, 1], seeds_voxel[:, 2], affine
    )
    seed_world = np.column_stack(seed_world)
    _, seed_idx = tree.query(seed_world, k=1)
    for i, idx in enumerate(seed_idx):
        A_sparse[i, idx] = True

    # Sanity checks
    sphere_sizes = np.asarray(A_sparse.sum(axis=1)).ravel()
    empty = np.nonzero(sphere_sizes == 0)[0]
    if len(empty):
        raise ValueError(f"Empty spheres: {empty}")
    if not allow_overlap and np.any(A_sparse.sum(axis=0) >= 2):
        raise ValueError("Overlap detected between spheres")

    return X, A_sparse.tocsr()

# ------------------------------------------------------------------------
# Shrinkage & whitening. From Walther 2016 an RSA toolbox descriptions 
# ------------------------------------------------------------------------
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
        #print("Warning: Shrunk covariance not positive definite. Adding epsilon to diagonal.")
        sigma_shrunk_reg = sigma_shrunk + epsilon * np.eye(n)
        # Try again with regularized version
        try:
          np.linalg.cholesky(sigma_shrunk_reg)
          sigma_shrunk = sigma_shrunk_reg # Use regularized if it works
        except np.linalg.LinAlgError:
          #print("Warning: Regularized shrunk covariance also not positive definite. Using prior for whitening.")
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

# ------------------------------------------------------------------------
# Process one sphere, and get distances bewteen adjacent trials
# ------------------------------------------------------------------------
def process_sphere(idx, tb_VxT, devs, A, n_trials):
    vox_inds = A[idx].indices # get the indices of the voxels in this sphere
    if len(vox_inds) == 0:
        return np.full(n_trials, np.nan), np.full(n_trials, np.nan)
    dev_roi = devs[:, vox_inds] # get the deviation patterns for this sphere for noise cov
    if dev_roi.shape[0] >= 2:
        W, _ = compute_shrunk_covariance_and_whitening_matrix(dev_roi, return_shrunk_cov=True) 
    else:
        W = np.eye(len(vox_inds))
    b_roi = tb_VxT[vox_inds, :] # get the betas for this sphere
    wb = W.dot(b_roi) # whitened betas
    data_tv = wb.T # transpose to get trials as rows
    #euc = np.full(n_trials, np.nan)
    corrd = np.full(n_trials, np.nan)
    #diffs = data_tv[1:] - data_tv[:-1]
    #euc[1:] = np.linalg.norm(diffs, axis=1)
    for t in range(1, n_trials):
        r = np.corrcoef(data_tv[t-1], data_tv[t])[0,1]
        corrd[t] = np.sqrt(2 * (1 - r)) # convert to distance 
    return corrd, _ #euc # not doing euc for now

# ------------------------------------------------------------------------
# Main searchlight adjacency RSA pipeline
# ------------------------------------------------------------------------
def main():
    subs = input()
    mask_dir = "/Shared/lss_kahwang_hpc/ROIs"
    GLM_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle"
    out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
    #os.makedirs(out_dir, exist_ok=True)
    ROI_mask = "Schaefer400+Morel+BG_2.5.nii.gz"
    mask_path = os.path.join(mask_dir, ROI_mask)
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    binary = (mask_data > 0).astype(np.int8)
    binary_mask_img = nilearn.image.new_img_like(mask_nii, binary)
    ijk = np.column_stack(np.where(binary == 1))
    seeds = np.column_stack( coord_transform(ijk[:,0], ijk[:,1], ijk[:,2], mask_nii.affine) )
    radius = 9.5
    allow_overlap = True
    n_jobs = 112
    for s in [subs]:
        print(f"\n=== Subject {s} ===")
        subj = os.path.join(GLM_dir, f"sub-{s}")
        tb_file = os.path.join(subj, f"{s}_TrialBetas_resRT.nii.gz")
        if not os.path.exists(tb_file):
            print("Missing TrialBetas:", tb_file); continue
        tb_nii = nib.load(tb_file)
        n_trials = tb_nii.shape[3]
        tb_mat = masking.apply_mask(tb_nii, binary_mask_img)
        tb_VxT = tb_mat.T # voxels x trials
        mean_beta = tb_VxT.mean(axis=1, keepdims=True)
        devs = (tb_VxT - mean_beta).T # get deviations for noise covariance
        X, A = new_apply_mask_and_get_affinity(
            seeds=seeds,
            niimg=tb_nii,
            radius=radius,
            allow_overlap=allow_overlap,
            mask_img=binary_mask_img
        )
        n_seeds = A.shape[0]
        print(f"  # spheres: {n_seeds}")
        results = Parallel(n_jobs=n_jobs, verbose=2)(
            delayed(process_sphere)(i, tb_VxT, devs, A, n_trials)
            for i in range(n_seeds)
        )
        corr_adj = np.vstack([c for c,_ in results])
        #euc_adj = np.vstack([e for _,e in results])
        np.save(os.path.join(out_dir, f"{s}_sl_adjcorr.npy"), corr_adj)
        #np.save(os.path.join(out_dir, f"{s}_sl_adjeuc.npy"),  euc_adj)
        print(f"Saved shapes: corr {corr_adj.shape}")

if __name__ == "__main__":
    start = datetime.now()
    print("Start:", start)
    main()
    print("End:", datetime.now(), "Duration:", datetime.now() - start)

