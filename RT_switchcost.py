# code to analyze the source of RT switch cost, does it scale with neural distance?
import numpy as np
import pandas as pd
from scipy.stats import zscore
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
from nilearn import plotting
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib

def write_stats_to_vol_yeo_template_nifti(graph_metric, fn, roisize = 418):
	'''short hand to write vol based nifti file of the stats
	, voxels in each parcel will be replaced with the stats'''


	vol_template = nib.load('/mnt/nfs/lss/lss_kahwang_hpc/ROIs/Schaefer400+Morel+BG_2.5.nii.gz')
	v_data = vol_template.get_fdata()
	graph_data = np.zeros((np.shape(v_data)))

	for i in np.arange(roisize):
		graph_data[v_data == i+1] = graph_metric[i]

	new_nii = nib.Nifti1Image(graph_data, affine = vol_template.affine, header = vol_template.header)
	nib.save(new_nii, fn) 
	return new_nii

#load behavioral data
data_dir =  '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/coefs/'

df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv")
subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']
#s = 10001

results = []
accu_results = []
for s in subjects:
    #load neural correlation distance
    data = np.load(data_dir+ "%s_whole_brain_coef.npy" %s)  # this is in ROI by trial by trial

    num_trials = data.shape[2]
    roi_num = data.shape[0]
    RTs = df.loc[df['sub']==s]['rt'].values #raw RT
    zRTs = zscore(RTs, nan_policy="omit") #normalize RT to zscore
    responses = df.loc[df['sub']==s]['Subject_Respo'].values
    condition = df.loc[df['sub']==s]['Trial_type'].values
    accu = df.loc[df['sub']==s]['trial_Corr'].values #accuracy
    EDS = 1*(condition=="EDS")
    IDS = 1*(condition=="IDS")
    Stay = 1*(condition=="Stay")
    cues = df.loc[df['sub']==s].cue.values  #need to do cue repeat
    run_breaks = np.arange(0,num_trials,51)
    
    s_results=[]
    accu_sresults=[]
    for roi in np.arange(roi_num):
        
        #dervitaive variables
        ds = np.zeros(num_trials)
        dRTs = np.zeros(num_trials)
        resp = np.zeros(num_trials) #response repeat
        cue = np.zeros(num_trials) #cue repeat

        # cal trial by trial derivatives
        for t1 in np.arange(num_trials-1):
            t2 = t1+1
            ds[t2] = data[roi, t1, t2]
            dRTs[t2] = zRTs[t2] - zRTs[t1] # if zscore then take difference looks quite gaussian 
            if responses[t2] == responses[t1]:
                 resp[t2]=1
            if cues[t2] == cues[t1]:
                 cue[t2]=1
        ds = np.sqrt(2*(1-abs(ds)))   #correlation distance
        
        rdf = pd.DataFrame({"distance": ds})
        rdf['dRT'] = dRTs
        rdf['RT'] = np.log(RTs)
        rdf['response'] = resp
        rdf['condition'] = condition
        rdf['cue'] = cue
        rdf['EDS'] = EDS
        rdf['IDS'] = IDS
        rdf['Stay'] = Stay
        rdf['accu'] = accu
        rdf = rdf.drop(run_breaks)
        rdf = rdf.dropna()
        
        #regression of RT
        model = smf.ols(formula = "dRT ~ 1 + cue + response + distance:EDS + distance:IDS + distance:Stay ", data=rdf).fit()
        mdf = pd.read_html(model.summary().tables[1].as_html(), header=0)[0].rename(columns={'P>|t|': 'pvalue', 'Unnamed: 0': "parameter"})
        mdf['ROI'] = roi+ 1
        mdf['subject'] = s
        s_results.append(mdf)

        #logistic regression of accuracy
        # model = smf.logit(formula = "accu ~ 1 + cue + response + distance:EDS + distance:IDS + distance:Stay ", data=rdf).fit()
        # mdf = pd.read_html(model.summary().tables[1].as_html(), header=0)[0].rename(columns={'P>|t|': 'pvalue', 'Unnamed: 0': "parameter"})
        # mdf['ROI'] = roi+ 1
        # mdf['subject'] = s
        # accu_sresults.append(mdf)

    results.append(pd.concat(s_results))
    #accu_results.append(pd.concat(accu_sresults))

results_df = pd.concat(results, ignore_index=True)
results_df.to_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/rt_stats.csv')
#accu_results_df = pd.concat(accu_results, ignore_index=True)
#accu_results_df.to_csv('/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/accu_stats.csv')

parameters = results_df['parameter'].unique()

stats = []
for r in results_df.ROI.unique():
    for parameter in parameters: 
        tdf = results_df.loc[(results_df['ROI']==r) & (results_df['parameter']==parameter)]
        t_stat, p_value = ttest_1samp(tdf['coef'], 0)
        ttdf = pd.DataFrame({
            't-statistic': [t_stat],
            'p-value': [p_value],
            'mean': [np.nanmean(tdf['coef'])],
            'ROI': r,
            'model': parameter,
            })
        stats.append(ttdf)    
stats_df = pd.concat(stats, ignore_index=True)

# FDR correction
stats_df['q'] = fdrcorrection(stats_df['p-value'])[1]

# create thresholded nii images      
thres_niis = {}
for m in parameters:
    fn = '/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/niis/%s_switch_cost_t-statistic_thresholded.nii.gz' %m
    metric = stats_df.loc[stats_df['model']== m]['t-statistic'].values
    mask = stats_df.loc[stats_df['model']== m]['q'].values < .05
    mask[np.argsort(metric)[:-15]]=0
    thres_niis[m] = write_stats_to_vol_yeo_template_nifti(metric * mask, fn, roisize = 400)
    

plotting.plot_glass_brain(thres_niis['distance:EDS'], display_mode='lzry', plot_abs=False,
                          title='', threshold=2)
plt.show()
