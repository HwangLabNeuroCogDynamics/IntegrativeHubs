## create RSA models
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## create model regressors, run on thalamege. 

df = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv")

subjects = pd.read_csv("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']

for s in subjects:
    sdf = df.loc[df['sub'] == s]
    sdf = sdf.sort_values(by=['block', 'trial_n'], ascending=[True, True])
    num_trials = len(sdf)

    context_model = np.zeros((num_trials, num_trials))
    color_model = np.zeros((num_trials, num_trials))
    shape_model = np.zeros((num_trials, num_trials))
    task_model = np.zeros((num_trials, num_trials))
    response_model = np.zeros((num_trials, num_trials))
    stim_model = np.zeros((num_trials, num_trials))
    feature_model = np.zeros((num_trials, num_trials))

    for i in np.arange(num_trials):
        context_model[i,:] = 1*(sdf['Texture'].values == sdf.iloc[i]['Texture'])
        color_model[i,:] = 1*(sdf['Color'].values == sdf.iloc[i]['Color'])
        shape_model[i,:] = 1*(sdf['Shape'].values == sdf.iloc[i]['Shape'])
        task_model[i,:] = 1*(sdf['Task'].values == sdf.iloc[i]['Task'])
        response_model[i,:] = 1*(sdf['Subject_Respo'].values == sdf.iloc[i]['Subject_Respo'])
        stim_model[i,:] = 1*(sdf['pic'].values == sdf.iloc[i]['pic'])

        if isinstance(sdf.iloc[i].version, str):
            if sdf.iloc[i]['Texture'] == 'Donut':
                rel_feature = 'Shape'
                feature_model[i,:] = 1*(sdf[rel_feature].values == sdf.iloc[i][rel_feature])  
                
            if sdf.iloc[i]['Texture'] == 'Filled':
                rel_feature = 'Color'
                feature_model[i,:] = 1*(sdf[rel_feature].values == sdf.iloc[i][rel_feature]) 

        else:
            if sdf.iloc[i]['Texture'] == 'Donut':
                rel_feature = 'Color'
                feature_model[i,:] = 1*(sdf[rel_feature].values == sdf.iloc[i][rel_feature])  
                #& (sdf['Texture'].values == 'Donut'))
                
            if sdf.iloc[i]['Texture'] == 'Filled':
                rel_feature = 'Shape'
                feature_model[i,:] = 1*(sdf[rel_feature].values == sdf.iloc[i][rel_feature]) 
                #& (sdf['Texture'].values == 'Filled'))


    np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/models/%s_context_model.npy" %s, context_model)
    np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/models/%s_color_model.npy" %s, color_model)
    np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/models/%s_shape_model.npy" %s, shape_model)
    np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/models/%s_task_model.npy" %s, task_model)
    np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/models/%s_response_model.npy" %s, response_model)
    np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/models/%s_stim_model.npy" %s, stim_model)
    np.save("/mnt/nfs/lss/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/models/%s_feature_model.npy" %s, feature_model)