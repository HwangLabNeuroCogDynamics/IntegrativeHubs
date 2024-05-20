import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
import os
import nibabel as nib
import glob

################################
## Calculate PC for the thalhi dataset
################################

def generate_correlation_mat(x, y):
	"""Correlate each n with each m.

	Parameters
	----------
	x : np.array
	  Shape N X T.

	y : np.array
	  Shape M X T.

	Returns
	-------
	np.array
	  N X M array in which each element is a correlation coefficient.

	"""
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
		raise ValueError('x and y must ' +
						 'have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x,
				 y.T) - n * np.dot(mu_x[:, np.newaxis],
								  mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def write_graph_to_vol_yeo_template_nifti(graph_metric, fn, roisize=400):
	'''short hand to write vol based nifti file of the graph metrics
	, voxels in each parcel will be replaced with the graph metric'''


	vol_template = nib.load('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')
	v_data = vol_template.get_fdata()
	graph_data = np.zeros((np.shape(v_data)))

	for i in np.arange(roisize):
		graph_data[v_data == i+1] = graph_metric[i]

	new_nii = nib.Nifti1Image(graph_data, affine = vol_template.affine, header = vol_template.header)
	nib.save(new_nii, fn)



sub_df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/censor_percentage_by_subjs.csv")
del_subs = []
for ij, sub in enumerate(sub_df['sub']):
	isExist = os.path.exists("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-%s/sub-%s_FIRmodel_errts_8cues2resp.nii.gz" %(sub, sub))
	if not isExist:
		del_subs.append(ij)
sub_df = sub_df.drop(del_subs)
sub_df.to_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")


if __name__ == "__main__":

	### global variables, masks, files, etc.
	# load masks
	sub_df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")
	Schaefer400_mask = nib.load('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')
	cortex_masker = NiftiLabelsMasker(labels_img='/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz', standardize=False)
	Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')


	## thalamus stuff for the future
	# thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
	# thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_fdata()
	# thalamus_mask_data = thalamus_mask_data>0
	# thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data)

	thresholds = [86,87,88,89,90,91,92,93,94,95,96,97,98,99]
	pc_vectors = np.zeros((len(Schaeffer_CI),len(sub_df), len(thresholds))) #dimension is ROI by sub by threshold 

	for ij in np.arange(len(sub_df)):
		# load files
		sub = sub_df.loc[ij, 'sub'] 
		resid_fn = '/home/kahwang/LSS_data/ThalHi/3dDeconvolve_fdpt4/sub-%s/sub-%s_FIRmodel_errts_8cues2resp.nii.gz' %(sub,sub)
		#resid_files = glob.glob(resid_fn)

		functional_data = nib.load(resid_fn)
		cortex_ts = cortex_masker.fit_transform(functional_data)

		#get rid of sensored timepoints
		cortex_ts = cortex_ts[np.mean(cortex_ts,axis=1)!=0,:]
		corr_mat = generate_correlation_mat(cortex_ts.T, cortex_ts.T)

		## now calculate PC..
		for it, t in enumerate(thresholds):
			temp_mat = corr_mat.copy()
			temp_mat[temp_mat<np.percentile(temp_mat, t)] = 0
			fc_sum = np.sum(temp_mat, axis=1)
			kis = np.zeros(np.shape(fc_sum))

			for ci in np.unique(Schaeffer_CI):
				kis = kis + np.square(np.sum(temp_mat[:,np.where(Schaeffer_CI==ci)[0]], axis=1) / fc_sum)

			pc_vectors[:,ij, it] = 1-kis

	np.save("/home/kahwang/bin/IntegrativeHubs/data/pc_vectors.npy", pc_vectors)
	np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/pc_vectors.npy", pc_vectors) #should we save it somewhere else?


	#organize data frame
	pc_df = pd.DataFrame()

	for ij in np.arange(len(sub_df)):
		
		sdf = pd.DataFrame()
		# load files
		sub = sub_df.loc[ij, 'sub'] 
		sdf.loc[:,'PC'] = np.mean(pc_vectors[:,ij,:], axis=1)
		sdf.loc[:,'Sub'] = sub
		sdf.loc[:,'ROI'] = np.arange(pc_vectors.shape[0])+1
		pc_df = pc_df.append(sdf)
	pc_df.to_csv("/home/kahwang/bin/IntegrativeHubs/data/pc_vectors.csv")
	pc_df.to_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/pc_vectors.csv")


	#write nifit to check visually
	write_graph_to_vol_yeo_template_nifti(np.mean(pc_vectors, axis=(1,2)), "/home/kahwang/bin/IntegrativeHubs/data/pc_vectors.nii.gz", roisize=400)

#end of line
