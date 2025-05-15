##### write distnace to 3dDeconvolve 1d format. Need to use older version of pandas because the append method
import numpy as np
import pandas as pd

func_path = "/Shared/lss_kahwang_hpc/data/ThalHi/"
decon_path = "/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/"
df = pd.read_csv(func_path + "ThalHi_MRI_2020_RTs.csv")

subjects = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/usable_subjs.csv")['sub']
#df['sub'].unique().astype("int")

for sub in subjects:
    try:
        eds=pd.DataFrame()
        ids=pd.DataFrame()
        stay=pd.DataFrame()
        # errors=pd.DataFrame()
        # all_trials=pd.DataFrame()
        rt=pd.DataFrame()
        runs=[1,2,3,4,5,6,7,8]
        num_trials = len((df.loc[(df['sub'] == sub)]))
        run_breaks = np.arange(0,num_trials,51)
        #print(sub)
        df.loc[(df['sub'] == sub), 'switch_distance'] = np.loadtxt(decon_path + "sub-%s/switch_distance.txt" %sub)
        df.loc[(df['sub'] == sub), 'EDS_switch_distance'] = np.loadtxt(decon_path + "sub-%s/EDS_switch_distance.txt" %sub)
        df.loc[(df['sub'] == sub), 'IDS_switch_distance'] = np.loadtxt(decon_path + "sub-%s/IDS_switch_distance.txt" %sub)
        df.loc[(df['sub'] == sub), 'Stay_switch_distance'] = np.loadtxt(decon_path + "sub-%s/Stay_switch_distance.txt" %sub)
        eds_distance=pd.DataFrame()
        ids_distance=pd.DataFrame()
        stay_distance=pd.DataFrame()        
        switch_distance = pd.DataFrame()
        
        for r in runs:
            eds = eds.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['Trial_type'] == 'EDS') & (df['trial_Corr']==1)].Time_Since_Run_Cue_Prez.reset_index(drop=True))
            ids = ids.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['Trial_type'] == 'IDS') & (df['trial_Corr']==1)].Time_Since_Run_Cue_Prez.reset_index(drop=True))
            stay = stay.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['Trial_type'] == 'Stay') & (df['trial_Corr']==1)].Time_Since_Run_Cue_Prez.reset_index(drop=True))
            #errors = errors.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['trial_Corr']==0)].Time_Since_Run_Cue_Prez.reset_index(drop=True)) # not sure we want to model this given the low number of error trials
            #all_trials = all_trials.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['trial_Corr']==1)].Time_Since_Run_Cue_Prez.reset_index(drop=True))
            #rt = rt.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['trial_Corr']==1)].rt.reset_index(drop=True))
            eds_distance = eds_distance.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['Trial_type'] == 'EDS') & (df['trial_Corr']==1)].EDS_switch_distance.reset_index(drop=True))
            ids_distance = ids_distance.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['Trial_type'] == 'IDS') & (df['trial_Corr']==1)].IDS_switch_distance.reset_index(drop=True))
            stay_distance = stay_distance.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['Trial_type'] == 'Stay') & (df['trial_Corr']==1)].Stay_switch_distance.reset_index(drop=True))
            switch_distance = switch_distance.append(df.loc[(df['sub'] == sub) & (df['block'] == r) & (df['trial_Corr']==1)].switch_distance.reset_index(drop=True))

        #np.savetxt(decon_path + 'sub-{}/EDS.acc.1D'.format(sub),eds.to_numpy(),fmt='%1.4f') 
        #np.savetxt(decon_path + 'sub-{}/IDS.acc.1D'.format(sub),ids.to_numpy(),fmt='%1.4f')
        #np.savetxt(decon_path + 'sub-{}/Stay.acc.1D'.format(sub),stay.to_numpy(),fmt='%1.4f')
        #np.savetxt(decon_path + 'sub-{}/Errors.1D'.format(sub),errors.to_numpy(),fmt='%1.4f')
        #np.savetxt(decon_path + 'sub-{}/All_Trials.1D'.format(sub),all_trials.to_numpy(),fmt='%1.4f')
        #np.savetxt(decon_path + 'sub-{}/RT.1D'.format(sub),rt.to_numpy(),fmt='%1.4f')
        np.savetxt(decon_path + 'sub-{}/EDS_distance.1D'.format(sub),eds_distance.to_numpy(),fmt='%1.4f')
        np.savetxt(decon_path + 'sub-{}/IDS_distance.1D'.format(sub),ids_distance.to_numpy(),fmt='%1.4f')
        np.savetxt(decon_path + 'sub-{}/Stay_distance.1D'.format(sub),stay_distance.to_numpy(),fmt='%1.4f')
        np.savetxt(decon_path + 'sub-{}/Switch_distance.1D'.format(sub),switch_distance.to_numpy(),fmt='%1.4f')

    except:
        print(sub)
    ## do this to get rid of nans sed -i "s/nan//g" RT.1D
    # sed -i "s/nan//g" All_Trials.1D


# write out RTs as regressors
for sub in subjects:
    try:
        sub_Df = df.loc[(df['sub'] == sub)]
        sub_Df = sub_Df.sort_values(by =['block', 'trial_n'])
        rt = sub_Df['RT_z'].to_numpy()
        rt[np.isnan(rt)] = np.nanmean(rt)
        np.savetxt("/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/sub-{}/RT.1D". format(sub), rt, fmt='%1.4f')
    except:
        print(sub)