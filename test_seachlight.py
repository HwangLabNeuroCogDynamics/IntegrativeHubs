# script to see if we can speed up nilearn's searchlight process
import warnings
import numpy as np
from scipy import sparse
from sklearn import neighbors
from nilearn import image, masking
from nilearn._utils.niimg_conversions import (
    check_niimg_3d,
    check_niimg_4d,
    safe_get_data,
)
from nilearn.image.resampling import coord_transform
from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity
import datetime


def new_apply_mask_and_get_affinity( seeds, niimg, radius, allow_overlap, mask_img=None ):
    seeds = np.asarray(seeds)
    if seeds.ndim == 1:
        seeds = seeds[np.newaxis, :]

    # Compute world coordinates of all in-mask voxels.
    if niimg is None:
        mask, affine = masking.load_mask_img(mask_img)
        mask_coords = np.column_stack(np.where(mask))
        X = None
    elif mask_img is not None:
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
    else:
        affine = niimg.affine
        if np.isnan(np.sum(safe_get_data(niimg))):
            warnings.warn(
                "The imgs you have fed into fit_transform() contains NaN "
                "values which will be converted to zeroes."
            )
            X = safe_get_data(niimg, True).reshape([-1, niimg.shape[3]]).T
        else:
            X = safe_get_data(niimg).reshape([-1, niimg.shape[3]]).T
        mask_coords = np.column_stack(np.indices(niimg.shape[:3]).reshape(3, -1).T)

    # Transform all mask coordinates to world space at once
    mask_coords_world = image.resampling.coord_transform(
        mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2], affine
    )
    mask_coords_world = np.column_stack(mask_coords_world)

    # Find nearest voxel for each seed
    inv_affine = np.linalg.inv(affine)
    seeds_voxel = np.dot(np.column_stack([seeds, np.ones(len(seeds))]), inv_affine.T)[:, :3]
    nearest_voxels = np.round(seeds_voxel).astype(int)
    
    # Create a dictionary for fast lookup of coordinate indices
    coord_dict = {tuple(coord): idx for idx, coord in enumerate(mask_coords)}
    nearests = [coord_dict.get(tuple(coord), None) for coord in nearest_voxels]

    # Use KDTree for faster radius queries
    tree = neighbors.KDTree(mask_coords_world)
    A = tree.query_radius(seeds, r=radius)
    
    # Convert to sparse matrix
    n_seeds = len(seeds)
    n_voxels = len(mask_coords)
    A_sparse = sparse.lil_matrix((n_seeds, n_voxels), dtype=bool)
    
    for i, (neighbors_idx, nearest) in enumerate(zip(A, nearests)):
        if neighbors_idx.size > 0:
            A_sparse[i, neighbors_idx] = True
        if nearest is not None:
            A_sparse[i, nearest] = True

    # Include seed voxels if not already included
    seed_voxel_world = image.resampling.coord_transform(
        seeds_voxel[:, 0], seeds_voxel[:, 1], seeds_voxel[:, 2], affine
    )
    seed_voxel_world = np.column_stack(seed_voxel_world)
    _, seed_indices = tree.query(seed_voxel_world, k=1)
    for i, idx in enumerate(seed_indices):
        A_sparse[i, idx] = True

    sphere_sizes = np.asarray(A_sparse.sum(axis=1)).ravel()
    empty_spheres = np.nonzero(sphere_sizes == 0)[0]
    if len(empty_spheres) != 0:
        raise ValueError(f"These spheres are empty: {empty_spheres}")

    if not allow_overlap and np.any(A_sparse.sum(axis=0) >= 2):
        raise ValueError("Overlap detected between spheres")

    return X, A_sparse



### now let us do some testing...
subject = '10001'
func_path = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/sub-%s/" %subject
mask_path = "/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-%s/combined_mask.nii.gz" %subject
mask_img = image.load_img(mask_path)

# Get the seeds
process_mask_img = mask_img

# Compute world coordinates of the seeds
process_mask, process_mask_affine = masking.load_mask_img( process_mask_img )
process_mask_coords = np.where(process_mask != 0)
process_mask_coords = coord_transform(
    process_mask_coords[0],
    process_mask_coords[1],
    process_mask_coords[2],
    process_mask_affine,
)
process_mask_coords = np.asarray(process_mask_coords).T

# now = datetime.datetime.now()
# print(now)
# X_old, A_old = _apply_mask_and_get_affinity( process_mask_coords, mask_img, 6, True, mask_img=mask_img)
# now = datetime.datetime.now()
# print(now)


# now try the new function
now = datetime.datetime.now()
print(now)
X_new, A_new = new_apply_mask_and_get_affinity( process_mask_coords, mask_img, 6, True, mask_img=mask_img)
now = datetime.datetime.now()
print(now)


def simple_apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap, mask_img=None):
    # Construct searchlight spheres (affinity matrix) around seed points, adapted from Nilearn
    # Input parameters:
    #   seeds       : array-like of shape (n_seeds, 3), world-space (x,y,z) coordinates in mm
    #   niimg       : 4D Nifti image or filename of the functional data (for shape reference) 
    #                 (not actually used here, but kept for compatibility)
    #   radius      : float, sphere radius in mm around each seed
    #   allow_overlap: bool, if False, raises an error when spheres overlap
    #   mask_img    : 3D Nifti image or filename of a **binary** mask (non-zero voxels define searchlight domain)
    #
    # Returns:
    #   X           : None placeholder (to match API), can be ignored
    #   A_sparse    : scipy.sparse CSR matrix, shape (n_seeds, n_voxels_in_mask), 
    #                 A_sparse[i, j] == True if voxel j is in sphere i
    # Binarized mask_img assumed
    mask, affine = masking.load_mask_img(mask_img)
    mask_coords = np.column_stack(np.where(mask != 0))
    mask_coords_world = np.column_stack(
        coord_transform(mask_coords[:,0],
                        mask_coords[:,1],
                        mask_coords[:,2],
                        affine)
    )
    tree = KDTree(mask_coords_world)
    neighbors_list = tree.query_radius(seeds, r=radius)

    n_seeds = len(seeds)
    n_vox   = mask_coords.shape[0]
    A_sparse = sparse.lil_matrix((n_seeds, n_vox), dtype=bool)

    inv_affine = np.linalg.inv(affine)
    seeds_vox = (
        np.column_stack([seeds, np.ones(n_seeds)]) @ inv_affine.T
    )[:,:3].round().astype(int)
    coord2idx = {tuple(c): i for i, c in enumerate(mask_coords)}

    for i, neigh in enumerate(neighbors_list):
        A_sparse.rows[i] = list(neigh)
        sv = tuple(seeds_vox[i])
        if sv in coord2idx:
            A_sparse.rows[i].append(coord2idx[sv])
        if not allow_overlap and len(A_sparse.rows[i]) != len(set(A_sparse.rows[i])):
            raise ValueError(f"Overlap detected in sphere {i}")

    return None, A_sparse.tocsr()

now = datetime.datetime.now()
print(now)
_, A_sim = simple_apply_mask_and_get_affinity( process_mask_coords, mask_img, 6, True, mask_img=mask_img)
now = datetime.datetime.now()
print(now)