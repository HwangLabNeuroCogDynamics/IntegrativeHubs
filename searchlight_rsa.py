####################################################################
# Searchlight adj trial neural distance script
# - Fast custom affinity function for spheres
# - Computes correlation distances between adjacent trials --- Neural Distance!
####################################################################

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
import numpy as np
import nibabel as nib
from datetime import datetime
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn import neighbors
from nilearn import image, masking
from nilearn._utils.niimg_conversions import check_niimg_3d, safe_get_data
from nilearn.image.resampling import coord_transform
from joblib import Parallel, delayed
import nilearn
import pandas as pd


# ======== Custom affinity function (adapted from nilearn) ========
def new_apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap, mask_img=None):
    """
    Build a sparse sphere-to-voxel affinity matrix.

    seeds         : array (n_seeds, 3) in world (mm) coordinates
    niimg         : Nifti1Image of 4D data (for affine) or None
    radius        : searchlight radius in mm
    allow_overlap : bool, raise error if overlap detected
    mask_img      : 3D binary mask (Nifti1Image or filename)

    Returns:
        X        : data matrix (unused here if niimg=None)
        A_sparse : sparse matrix (n_seeds x n_voxels) boolean
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
# ========================================


# ======== Shrinkage & whitening (Walther 2016, RSA toolbox) ========
def compute_shrunk_covariance_and_whitening_matrix(x, return_shrunk_cov=False):
    """
    Compute shrunk covariance matrix and its inverse square root (whitening).
    Optionally return the shrunk covariance itself.

    x : data matrix, shape (t, n) with t observations × n voxels
    """
    t, n = x.shape
    if t < 2:
        print(f"Warning: not enough samples ({t}). Returning identity.")
        identity_matrix = np.eye(n)
        if return_shrunk_cov:
            return identity_matrix, identity_matrix
        else:
            return identity_matrix

    # Demean data
    x_demeaned = x - x.mean(0)
    sample_cov = (1.0 / (t - 1)) * np.dot(x_demeaned.T, x_demeaned)
    prior = np.diag(np.diag(sample_cov))

    # Shrinkage coefficient
    num = np.sum((sample_cov - prior) ** 2)
    den = np.sum((x_demeaned**2 - np.diag(sample_cov)[np.newaxis, :]) ** 2) / t
    shrinkage = 1.0
    if den > 1e-10 and num > 1e-10:
        shrinkage = num / den
    shrinkage = np.clip(shrinkage, 0.0, 1.0)

    if t < n:
        print(f"Warning: samples ({t}) < voxels ({n}). Increasing shrinkage.")
        shrinkage = max(shrinkage, 1 - ((t - 1) / (n * 1.5)))
        shrinkage = np.clip(shrinkage, 0.0, 1.0)

    sigma_shrunk = shrinkage * prior + (1 - shrinkage) * sample_cov
    epsilon = 1e-7

    # Ensure positive definiteness
    try:
        np.linalg.cholesky(sigma_shrunk)
    except np.linalg.LinAlgError:
        sigma_shrunk = sigma_shrunk + epsilon * np.eye(n)

    # Whitening matrix
    whitening_matrix = np.eye(n)
    try:
        evals, evecs = np.linalg.eigh(sigma_shrunk)
        evals_floored = np.maximum(evals, epsilon)
        evals_sqrt_inv = 1.0 / np.sqrt(evals_floored)
        whitening_matrix = np.dot(evecs * evals_sqrt_inv, evecs.T)
    except np.linalg.LinAlgError as err:
        print(f"Whitening failed: {err}. Falling back to prior.")
        whitening_matrix = np.eye(n)

    if return_shrunk_cov:
        return whitening_matrix, sigma_shrunk
    else:
        return whitening_matrix


# ======== Function to process one sphere ========
def process_sphere(idx, tb_VxT, devs, A, n_trials, whiten="On"):
    vox_inds = A[idx].indices
    if len(vox_inds) == 0:
        return np.full(n_trials, np.nan), np.full(n_trials, np.nan)

    b_roi = tb_VxT[vox_inds, :]
    dev_roi = devs[:, vox_inds]

    if whiten == "On":
        if dev_roi.shape[0] >= 2:
            W, _ = compute_shrunk_covariance_and_whitening_matrix(dev_roi, return_shrunk_cov=True)
        else:
            W = np.eye(len(vox_inds))
        wb = W.dot(b_roi)
    else:
        wb = b_roi

    data_tv = wb.T
    corrd = np.full(n_trials, np.nan)
    for t in range(1, n_trials):
        r = np.corrcoef(data_tv[t - 1], data_tv[t])[0, 1]
        corrd[t] = np.sqrt(2 * (1 - r))
    return corrd, np.full(n_trials, np.nan)


# ======== Main searchlight adjacency neural distance pipeline ========
def run_func(subs, in_fn, out_fn, mnn):
    mask_dir = "/Shared/lss_kahwang_hpc/ROIs"
    GLM_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle"
    out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/searchlightRSA"
    # os.makedirs(out_dir, exist_ok=True)

    ROI_mask = "Schaefer400+Morel+BG_2.5.nii.gz"
    mask_path = os.path.join(mask_dir, ROI_mask)
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    binary = (mask_data > 0).astype(np.int8)
    binary_mask_img = nilearn.image.new_img_like(mask_nii, binary)

    ijk = np.column_stack(np.where(binary == 1))
    seeds = np.column_stack(
        coord_transform(ijk[:, 0], ijk[:, 1], ijk[:, 2], mask_nii.affine)
    )

    radius = 8
    allow_overlap = True
    n_jobs = 16

    for s in [subs]:
        print(f"\n=== Subject {s} ===")
        subj = os.path.join(GLM_dir, f"sub-{s}")
        tb_file = os.path.join(subj, "%s_%s.nii.gz" % (s, in_fn))
        if not os.path.exists(tb_file):
            print("Missing TrialBetas:", tb_file)
            continue

        tb_nii = nib.load(tb_file)
        n_trials = tb_nii.shape[3]
        tb_mat = masking.apply_mask(tb_nii, binary_mask_img)
        tb_VxT = tb_mat.T
        del tb_mat

        mean_beta = tb_VxT.mean(axis=1, keepdims=True)
        devs = (tb_VxT - mean_beta).T

        X, A = new_apply_mask_and_get_affinity(
            seeds=seeds,
            niimg=tb_nii,
            radius=radius,
            allow_overlap=allow_overlap,
            mask_img=binary_mask_img,
        )

        n_seeds = A.shape[0]
        print(f"  # spheres: {n_seeds}")

        results = Parallel(n_jobs=n_jobs, verbose=2)(
            delayed(process_sphere)(i, tb_VxT, devs, A, n_trials, whiten=mnn)
            for i in range(n_seeds)
        )

        corr_adj = np.vstack([c for c, _ in results])

        if mnn == "On":
            mnn_fn = "T"
        else:
            mnn_fn = "F"

        np.save(
            os.path.join(out_dir, "%s_sl_adjcorr_%s%s.npy" % (s, out_fn, mnn_fn)),
            corr_adj,
        )
        print(f"Saved shapes: corr {corr_adj.shape}")

if __name__ == "__main__":
    subs = input()
    start = datetime.now()
    print("Subject start:", start)
    run_func(subs, in_fn="TrialBetas_resRT", out_fn="resRTWT", mnn="On")
    print("End:", datetime.now(), "Duration:", datetime.now() - start)

#end of line
