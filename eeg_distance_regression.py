# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import mne
from joblib import Parallel, delayed
from itertools import product

#load behavioral data
ROOT = '/Shared/lss_kahwang_hpc/ThalHi_data/'
sub = input()  
fdf = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/ThalHi_EEG_zRTs_full2.csv")

# -------- whether probe, task, response, is a repeat from previous trial? Also code errors and stay CRs CSs
def add_prev_match(df, var, subject_col='sub', trial_col='trial_n', block_col=None ):
    """
    For each (subject[, block]) group in df, looks up the value of `var`
    at trial_n-1 and adds:
      - prev_<var>           : the previous-trial value of var (or NaN)
      - <var>_repeat     : "same"/"switch" comparing var vs prev_<var>
    
    Parameters
    ----------
    df : pandas.DataFrame
    var : str
        name of the column you want to compare (e.g. "pic" or "Subject_Respo")
    subject_col : str, default 'sub'
        name of the subject identifier column
    trial_col : str, default 'trial_n'
        name of the trial‐number column
    block_col : str or None
        if provided, also groups by this column so trial count resets each block
    
    Returns
    -------
    df : pandas.DataFrame
        the same DataFrame, with two new columns added in-place:
          * prev_<var>
          * <var>_match_prev
    """
    prev_col = f'prev_{var}'
    flag_col = f'{var}_repeat'
    
    # initialize
    df[prev_col] = pd.NA
    df[flag_col] = pd.NA
    
    # decide grouping keys
    group_cols = [subject_col]
    if block_col is not None:
        group_cols.append(block_col)
    
    # loop through groups
    for _, grp in df.groupby(group_cols):
        # mapping trial -> var for this group
        mapping = grp.set_index(trial_col)[var]
        # lookup trial-1
        prev_vals = grp[trial_col].sub(1).map(mapping)
        # write back
        df.loc[grp.index, prev_col] = prev_vals
        df.loc[grp.index, flag_col] = np.where( grp[var].eq(prev_vals), 'same', 'switch' )
    return df

df = fdf.loc[fdf['sub'] == int(sub)].sort_values(["block", "trial_n"]).reset_index(drop=True)

#response repeat
df = add_prev_match(df, var='Subject_Respo', subject_col='sub', trial_col='trial_n', block_col='block')
df['response_repeat'] = df['Subject_Respo_repeat']

#task repeat
df = add_prev_match(df, var='Task', subject_col='sub', trial_col='trial_n', block_col='block')
df['task_repeat'] = df['Task_repeat']

#probe repeat
df = add_prev_match(df, var='pic', subject_col='sub', trial_col='trial_n', block_col='block')
df['probe_repeat'] = df['pic_repeat']
df['Errors'] = 1*(df['trial_Corr']!=1)
for col in ["probe_repeat", "response_repeat", "task_repeat", "Trial_type", "prev_target_feature_match"]:
    df[col] = df[col].astype("category")    

# then need to figure out if any trials rejected...
this_sub_path=ROOT+ 'eeg_preproc_RespToReviewers/' +str(sub)
all_probe = mne.read_epochs(this_sub_path+'/probe events-epo.fif')
all_probe.baseline = None
meta_df = all_probe.metadata
meta_df = meta_df.sort_values(["block", "trial_n"]).reset_index(drop=True)

#check if number of trials are the same
num_trials = df.shape[0]
meta_df_trial_vector = (meta_df['block'] - 1) * 83 + meta_df['trial_n'] 
meta_df_trial_vector = meta_df_trial_vector.values
df_trial_vector = (df['block'] - 1) * 83 + df['trial_n']
df_trial_vector = df_trial_vector.values

if num_trials != meta_df.shape[0]:
    #find out trial vector
    trial_vector = np.intersect1d(df_trial_vector, meta_df_trial_vector).astype(int)
else:
    trial_vector = np.arange(num_trials)

# # need to also drop the next trial if the previous trial was dropped (distance calculated between trial N and N+1). For DF:
# df_drop_trials = np.array([i for i in range(num_trials) if i not in trial_vector])
# df_drop_trials = np.concatenate((df_drop_trials, df_drop_trials + 1))
# df_drop_trials = df_drop_trials[df_drop_trials<415]  # Ensure we don't go out of bounds
# if any(df_drop_trials):
#     mask = ~np.isin(df_trial_vector, df_drop_trials)
#     df_trial_vector = df_trial_vector[mask].astype(int)

# then need to drop the next trial in the coef_mat, which should be the same length as meta_df:
# coef_mat_drop_trials = np.array([i for i in range(coef_mat.shape[0]) if i not in trial_vector])
# coef_mat_drop_trials = coef_mat_drop_trials + 1 # trial N already dropped, so we drop the next trial
# coef_mat_drop_trials = coef_mat_drop_trials[coef_mat_drop_trials<415]  # Ensure we don't go out of bounds
# if any(coef_mat_drop_trials):
#     mask = ~np.isin(meta_df_trial_vector, coef_mat_drop_trials)
#     meta_df_trial_vector = meta_df_trial_vector[mask].astype(int)

# drop trials from coef_mat and df
# in the shape of trial by time by time, but here trial t is actually trial + 1 given it is a distance measure from the previous trial
# make sure it is distance from the "previous" trial
### looks like need quite a lot of memory for this 
coef_mat = np.load('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_coef_mat.npy')

# need to check that trial_vecotr is the same length as coef_mat
if len(trial_vector) != coef_mat.shape[0]:
    raise ValueError("Length of trial_vector does not match the number of trials in coef_mat. That means there are fewer trials in DF than the meta df in the coef_mat. Please check your data.")

# below no longer needed
# coef_mat[1:] = coef_mat[:-1]
# # Now set the first slice to NaNs:
# coef_mat[0].fill(np.nan)

#coef_mat = coef_mat[trial_vector,:,:]
df = df.loc[trial_vector, :].reset_index(drop=True)
df['trial_vector'] = trial_vector
num_tps = coef_mat.shape[1]  # Number of time points    

## now the regression 
def process_regression(t1, t2, coef_mat, df, model_syntax):
    """
    Process a single (t1, t2) iteration: build the DataFrame,
    drop run breaks and NaNs, run the regression, and return a list of results.
    """
    #build dataframe for regression
    df = df.copy()
    df['ds'] = coef_mat[:, t1, t2]
    df = df[(df['trial_n'] != 0) & (df['zRT'] <= 3) & (df['zRT'] > -3)].reset_index(drop=True) #drop first trial of each block and slow RT trials
    df['perceptual_change_c'] = df['perceptual_change'] - df['perceptual_change'].mean() #center regresor    
    
    # drop trials where to previous trial was dropped
    for i in range(1,len(df)):
        if df.loc[i,'trial_vector'] - df.loc[i-1, 'trial_vector'] != 1:
            df.loc[i, 'break'] = 1
    df = df.loc[df['break'] !=1]
    
    beta = {}
    tval = {}

    try:
        model  = smf.ols(model_syntax, data=df).fit()
        params = model.params

        # 1) raw betas & t-values
        beta.update(params.to_dict())
        tval.update(model.tvalues.to_dict())

        # 2) frequencies for Trial_type
        freq_tt = df['Trial_type'].value_counts(normalize=True)
        p_IDS   = float(freq_tt.get('IDS',  0))
        p_EDS   = float(freq_tt.get('EDS',  0))
        p_Stay  = float(freq_tt.get('Stay', 0))

        # frequencies for other factors
        p_resp = float(df['response_repeat']
                         .value_counts(normalize=True)
                         .get('switch', 0))
        p_task = float(df['task_repeat']
                         .value_counts(normalize=True)
                         .get('switch', 0))
        p_prev = float(df['prev_target_feature_match']
                         .value_counts(normalize=True)
                         .get('same_target_feature', 0))

        # 3) parameter‐name shortcuts
        name_pc       = "perceptual_change_c"
        name_eds_pc   = "C(Trial_type, Treatment(reference='IDS'))[T.EDS]:perceptual_change_c"
        name_stay_pc  = "C(Trial_type, Treatment(reference='IDS'))[T.Stay]:perceptual_change_c"

        name_eds_main = "C(Trial_type, Treatment(reference='IDS'))[T.EDS]"
        name_stay_main= "C(Trial_type, Treatment(reference='IDS'))[T.Stay]"

        name_rr   = "C(response_repeat)[T.switch]"
        name_rrtr = f"{name_rr}:C(task_repeat)[T.switch]"

        name_tr     = "C(task_repeat)[T.switch]"
        name_prev   = [n for n in params.index if "prev_target_feature_match" in n and "]" in n][0]
        name_trprev = f"{name_tr}:{name_prev}"

        # helper to add a contrast
        def add_mfx(key, L):
            ct = model.t_test(L)
            beta[key] = float(ct.effect)
            tval[key] = float(ct.tvalue)

        # 4) Marginal effects for perceptual_change_c
        # 4a) overall (weighted across IDS, EDS, Stay)
        L_pc = [
            p_IDS   if n==name_pc      else
            p_EDS   if n==name_eds_pc  else
            p_Stay  if n==name_stay_pc else
            0
            for n in params.index
        ]
        add_mfx('main_effect_perceptual_change_c', L_pc)

        # 4b) per-level slopes
        add_mfx('perceptual_change_for_IDS',
                [1 if n==name_pc else 0 for n in params.index])
        add_mfx('perceptual_change_for_EDS',
                [1 if n in (name_pc,name_eds_pc) else 0 for n in params.index])
        add_mfx('perceptual_change_for_Stay',
                [1 if n in (name_pc,name_stay_pc) else 0 for n in params.index])

        # 5) Marginal/main effects for other predictors
        # response_repeat overall
        L_rr = [
            1      if n==name_rr    else
            p_task if n==name_rrtr else
            0
            for n in params.index
        ]
        add_mfx('main_effect_response_repeat', L_rr)

        # task_repeat overall
        L_tr = [
            1      if n==name_tr      else
            p_resp if n==name_rrtr    else
            p_prev if n==name_trprev  else
            0
            for n in params.index
        ]
        add_mfx('main_effect_task_repeat', L_tr)

        # prev_target_feature_match overall
        L_prev = [
            1      if n==name_prev   else
            p_task if n==name_trprev else
            0
            for n in params.index
        ]
        add_mfx('main_effect_prev_target_feature_match', L_prev)

        # 6) Main effects of Trial_type contrasts
        add_mfx('main_effect_EDS_v_IDS',
                [1 if n==name_eds_main else 0 for n in params.index])
        add_mfx('main_effect_IDS_v_Stay',
                [1 if n==name_stay_main else 0 for n in params.index])

        # 7) Interaction of response_repeat × task_repeat
        add_mfx('interaction_response_repeat_x_task_repeat',
                [1 if n==name_rrtr else 0 for n in params.index])        
        
    except:
        pass #empty

    # Return the beta and t-value dictionaries for this (t1, t2) pair
    return beta, tval


model_syntax = (
    "ds ~ zRT + Errors + C(Trial_type, Treatment(reference='IDS')) * perceptual_change_c + "
    "C(response_repeat) * C(task_repeat) + "
    "C(prev_target_feature_match , Treatment(reference='switch_target_feature')) +"
    "C(probe_repeat)")

# Define your t1/t2 grid exactly as in your Parallel call
t1s = list(range(0, num_tps))
t2s = list(range(0, num_tps))
n1, n2 = len(t1s), len(t2s)

# 2 Run your Parallel (unchanged) to get flat lists of dicts
results = Parallel(n_jobs=16)( delayed(process_regression)(t1, t2, coef_mat, df, model_syntax) for t1, t2 in product(t1s, t2s) )
beta_dicts, tval_dicts = zip(*results)  # two tuples of length n1*n2

# Get the full set of parameter names (keys)
all_params = sorted(set().union(*beta_dicts))

# Pre-allocate arrays for each param
beta_maps = {p: np.full((n1, n2), np.nan, dtype=np.float32)
             for p in all_params}
tval_maps = {p: np.full((n1, n2), np.nan, dtype=np.float32)
             for p in all_params}

idx = 0
for i, t1 in enumerate(t1s):
    for j, t2 in enumerate(t2s):
        bdict = beta_dicts[idx]
        tdict = tval_dicts[idx]
        for p in all_params:
            if p in bdict:
                beta_maps[p][i, j]  = bdict[p]
                tval_maps[p][i, j]  = tdict[p]
        idx += 1

np.savez('/Shared/lss_kahwang_hpc/ThalHi_data/RSA/distance_results/' + str(sub) + '_model_betas_results.npz', **beta_maps)

