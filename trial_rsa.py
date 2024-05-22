import numpy as np
import pandas as pd
import os
import nibabel as nib
import glob
from datetime import datetime
import scipy
import nilearn
from nilearn.maskers import NiftiLabelsMasker


################################
## Calculate trial level RSA. use Argon
################################
included_subjects = input()


def compute_inv_shrunk_covariance(x):
    #see http://www.diedrichsenlab.org/pubs/Walther_Neuroimage_2016.pdf
    t,n = x.shape #t measurements by n voxels
    #demean
    x = x - x.mean(0)
    #compute covariance
    sample = (1.0/t) * np.dot(np.transpose(x),x)
    #copute prior
    prior = np.diag(np.diag(sample))
    #compute shrinkage
    d = 1.0/n * np.linalg.norm(sample - prior,ord = 'fro')**2
    y = np.square(x)
    r2 = 1.0/n/t**2 * np.sum(np.sum(np.dot(np.transpose(y),y)))- \
    1.0/n/t*np.sum(np.sum(np.square(sample)))
    #compute the estimator
    shrinkage = max(0,min(1,r2/d))
    sigma = shrinkage*prior + (1-shrinkage)*sample
    #compute the inverse
    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            inv_sigma = np.linalg.inv(prior) #univariate
        else:
            raise
    
    return inv_sigma


if __name__ == "__main__":

    now = datetime.now()
    print("Start time: ", now)
    
    #### setup
    ## relevant paths
    mask_dir = "/Shared/lss_kahwang_hpc/ROIs/"
    data_dr = "/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/"
    out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/"

    sub_df = pd.read_csv(data_dr + "usable_subjs.csv")

    # ROIs 
    ROI_mask = "Morel_2.5_labels.nii.gz"

    ##for now just testing this with Morel thalamus mask
    mask_file = os.path.join(mask_dir, ROI_mask)
    print("\n\nPulling mask file from ... ", mask_file)
    mask = nib.load(mask_file)
    mask_data = mask.get_fdata()
    print("mask shape ", mask_data.shape)
    
    #num_trials = 408
    num_subs = len(np.unique(sub_df['sub']))

    #sub_x_roi_x_coeff = np.zeros((num_subs, num_ROIs,num_trials,num_trials))

    for s in [included_subjects]:
        
        #load LSS data
        print("\nloading LSS for subject ", s)
        cur_lss_nii = nib.load( os.path.join(data_dr, ("sub-%s" %s),  ("cue.LSS.nii.gz")) )
        num_trials = int(cur_lss_nii.shape[3]/2)

        #load errts file
        print("\nloading errts for subjects ", s)
        errts = nib.load(os.path.join(data_dr, ("sub-%s" %s), ("sub-%s_FIRmodel_errts_8cues2resp.nii.gz" %s)))
        print("errts shape ",errts.shape)

        # check for all zeros in errts
        errts_data = errts.get_fdata()
        voxels_to_exclude = np.zeros((errts_data.shape[0], errts_data.shape[1], errts_data.shape[2])) # initialize 3D matrix of what voxels to exclude
        for x in range(errts_data.shape[0]):
            for y in range(errts_data.shape[1]):
                for z in range(errts_data.shape[2]):
                    if errts_data[x,y,z,:].sum()==0:
                        # voxel had 0 for all time points ... exclude from further analysis
                        voxels_to_exclude[x,y,z]=1
        print("A total of",voxels_to_exclude.sum(),"voxels will be EXCLUDED due to 0s for all time points")
        print(voxels_to_exclude.sum(),"voxels exluded out of",(errts_data.shape[0]*errts_data.shape[1]*errts_data.shape[2]),"total voxels")
        print((voxels_to_exclude.sum()/(errts_data.shape[0]*errts_data.shape[1]*errts_data.shape[2])),"proportion of voxels excluded\n")

        #Steph had a few masking operations that I don't fully understand
        # to make things faster, apply a cortcal mask to reduce roi mask to a vector
        # cortical_mask0 = nib.load(os.path.join(mask_dir,"CorticalMask_RSA_task-Quantum.nii.gz"))
        # cortical_data = cortical_mask0.get_fdata()

        # -- now modify mask to exclude zeros
        mask_binary=np.where(np.logical_and(mask_data>0,voxels_to_exclude==0),1.0,0) # make sure to also exclude voxels with all zeros
        print("number of usable voxels in current ROI mask:",mask_binary.sum(),"\n")
        ROI_binary_mask=nilearn.image.new_img_like(mask, mask_binary) # ...new_img_like(ref_niimg, data, ...)

        #apply mask to do something
        mask_vec = nilearn.masking.apply_mask(mask, ROI_binary_mask) # will be 1D (voxels)
        num_ROIs = len(np.unique(mask_vec))

        residual_2D = nilearn.masking.apply_mask(errts, ROI_binary_mask) # will be [time x voxels]
        #remove censored TRs, did Steph miss this step?
        residual_2D = residual_2D[np.sum(residual_2D, axis=1)!=0]
    
        # -- reduce LSS data to just amplitude information and then convert back to nii format
        # LSS data dim 4 set up like ...
        # [amplitude_trial_0, derivative_trial_0, amplitude_trial_1, derivative_trial_1, ...]
        cur_lss_data = cur_lss_nii.get_fdata()
        lss_data_idx = np.array( [np.arange(0,num_trials*2,2), np.arange(1,num_trials*2,2)] ) #.flatten() # so that all amplitude values are in the 1st half and all derivative values are in the 2nd half of the file
        cur_lss_data_amp = cur_lss_data[:, :, :, lss_data_idx[0]]
        cur_lss_amp_nii=nilearn.image.new_img_like(cur_lss_nii, cur_lss_data_amp) # convert back to nii
        print("lss shape ",cur_lss_amp_nii.shape)
        
        # -- mask LSS data to make it faster to loop through rois
        lss_2D = nilearn.masking.apply_mask(cur_lss_amp_nii, ROI_binary_mask) # will be [trials x voxels]
        #if lss_2D.shape[0] == num_trials:
        lss_2D = lss_2D.T # flip so that order is [voxels x trials]
        
        # now calculate the ROI by trial by trial similarity matrix.
        roi_x_coeff = np.zeros((num_ROIs,num_trials,num_trials))
        for i, roi in enumerate(np.unique(mask_vec)):
            print("\n\nworking on roi ", roi)
            # -- set up current roi binary mask
            # grab out current roi voxels
            cur_roi_inds = np.where(mask_vec==roi)[0]
            print("\nnumber of voxels in current ROI mask:", len(cur_roi_inds))
            if len(cur_roi_inds) == 0:
                print("current ROI has no voxels... saving [trials x trials] matrix as all NaN")
                roi_x_coeff[(roi-1),:,:] = np.ones((roi_x_coeff.shape[1],roi_x_coeff.shape[2]))*np.nan
            else:
                # -- prep for noise inflation correction
                r_masked = residual_2D[:,cur_roi_inds]

                # Sigma = (1/T) * (R_transpose * R)
                # bk* = bk * sigma^(-1/2)
                inv_sigma = compute_inv_shrunk_covariance(r_masked) # eq. 4 from paper JJ sent, plus inverse (part of eq. 6)
                inv_sigma = scipy.linalg.fractional_matrix_power(inv_sigma,0.5) # we already took inverse, so just sqrt now (part of eq. 6)
                print("\ninv_sigma size: ", inv_sigma.shape) # inv_sigma size should be [num_voxels x num_voxels]
                
                # -- apply mask to get voxels x trials matrix
                voxels_x_trials = lss_2D[cur_roi_inds,:] # should have flipped dims above so that lss_2D is [voxels x trials]
                # if voxels_x_trials.shape[0] == num_trials:
                #     voxels_x_trials = voxels_x_trials.T
                print("voxels by trials size: ", voxels_x_trials.shape)
                
                # eq. 6 ... just mult inv_sigma by the beta estimate matrix 
                voxels_x_trials_corrected = np.dot(inv_sigma,voxels_x_trials) # matrix multiplication 
                #print("voxels_x_trials: ",voxels_x_trials[:5,:8], "voxels_x_trials_corrected: ",voxels_x_trials_corrected[:5,:8])
                
                # -- now calculate correlation coef
                coeff_mat = np.corrcoef(voxels_x_trials_corrected, rowvar=False) # use pearson correlation
                print("coeff matrix size: ", coeff_mat.shape)
                #print("coeff matrix: ", coeff_mat[:5,40:45])
                # ax = sns.heatmap(coeff_mat, linewidth=0.5, cmap="Greens", vmin=0, vmax=1)
                # plt.title("coeff matrix", fontsize=20)
                # plt.show()
                # plt.close()

                # -- add coefficient matrix to ROI by coefficient matrix array
                roi_x_coeff[i,:,:] = coeff_mat

        #sub_x_roi_x_coeff[j,:,:,:] = roi_x_coeff
        #save output
        np.save(out_dir+"%s_morel_coef.npy" %s, roi_x_coeff)
        
        
        ### the next step is regression onto RSA models...

            

    now = datetime.now()
    print("End time: ", now)



# end