## now that we have single trial betas from GLMsingle, let us try decoding.
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import new_img_like, load_img, get_data, index_img
import nibabel as nib
from sklearn.model_selection import KFold
import nilearn.decoding
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR, SVC, LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
import nilearn
import datetime
from nilearn.maskers import NiftiMasker

#subject = input()
subject='10001'
func_path = "/Shared/lss_kahwang_hpc/data/ThalHi/GLMsingle/sub-%s/" %subject

# figure out which version the subject did, DCFS or DSFC
version_info = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/Version_Info.csv")
try:
    version = version_info.loc[version_info['sub'] == int(subject), 'version'].values[0]
except:
    version = 'DCFS'

# load the GLMsingle outputs
betas = nib.load(func_path +"%s_TrialBetas.nii.gz" %subject)
n_trials = betas.shape[3]

# load the design matrix
df = pd.read_csv("/Shared/lss_kahwang_hpc/data/ThalHi/ThalHi_MRI_2020_RTs.csv")
sub_df = df.loc[df['sub'] == int(subject), :]
# reorganize the design matrix by run then trials
sub_df = sub_df.sort_values(by=['block', 'trial_n'])

# test if number of trials is correct
if len(sub_df) != n_trials:
    print("number of trials not correct")
    raise ValueError("number of trials not correct")

### now make the Y 
context_Y = []
for cue in sub_df['cue']:
    if cue in ["dcb", "dcr", "dpb", "dpr", ]:
        context_Y.append('donut')
    if cue in ["fcb", "fcr", "fpb", "fpr"]:
        context_Y.append('fill')

color_Y = []
for cue in sub_df['cue']:
    if cue in ["dcb", "fcb", "dpb", "fpb", ]:
        color_Y.append('blue')
    if cue in ["dcr", "dpr", "fcr", "fpr"]:
        color_Y.append('red')

shape_Y = []
for cue in sub_df['cue']:
    if cue in ["dcb", "dcr", "fcb", "fcr", ]:
        shape_Y.append('circle')
    if cue in ["dpb", "dpr", "fpb", "fpr"]:
        shape_Y.append('polygon')

#task
task_Y = sub_df['Task'].values.tolist()

# feature is more complicated because of swaped design
feature_Y = []
if version == 'DCFS':
    feature_Y = []
    for cue in sub_df['cue']:
        if cue in ["dcb", "dpb"]:
            feature_Y.append('blue')
        if cue in ["dcr", "dpr" ]:
            feature_Y.append('red')
        if cue in ["fcr", "fcb"]:
            feature_Y.append('circle')
        if cue in ["fpr", "fpb" ]:
            feature_Y.append('polygon')

if version == 'DSFC':
    feature_Y = []
    for cue in sub_df['cue']:
        if cue in ["dcb", "dcr"]:
            feature_Y.append('circle')
        if cue in ["dpb", "dpr" ]:
            feature_Y.append('polygon')
        if cue in ["fcr", "fpr"]:
            feature_Y.append('red')
        if cue in ["fcb", "fpb"  ]:
            feature_Y.append('blue')

runs = sub_df['block'].values.tolist()

# now load the brain mask for decoding. As part of GLMsingle script, the mask was created in the 3dDeconvolve directory
mask_path = "/Shared/lss_kahwang_hpc/data/ThalHi/3dDeconvolve_fdpt4/sub-%s/" %subject
mask = nib.load (mask_path + "combined_mask.nii.gz")
          
### searchlight
now = datetime.datetime.now()
print(now)

### below is the real deal
mean_fmri = nilearn.image.mean_img(betas)
pipeline = make_pipeline(StandardScaler(), LogisticRegression('l2', solver='liblinear')) #ridge regression, downweight but no feature selection

try:
    searchlight_context = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_context.fit(betas, context_Y, runs)
    context_score_img = new_img_like(mean_fmri, searchlight_context.scores_)
    context_score_img.to_filename(func_path + "%s_context_logistic_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("context broke")
    pass

try:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression('l2', solver='liblinear')) 
    searchlight_color = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_color.fit(betas, color_Y, runs)
    color_score_img = new_img_like(mean_fmri, searchlight_color.scores_)
    color_score_img.to_filename(func_path + "%s_color_logistic_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("color broke")
    pass

try:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression('l2', solver='liblinear')) 
    searchlight_shape = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_shape.fit(betas, shape_Y, runs)
    shape_score_img = new_img_like(mean_fmri, searchlight_shape.scores_)
    shape_score_img.to_filename(func_path + "%s_shape_logistic_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("shape broke")
    pass

try:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression('l2', solver='liblinear')) 
    searchlight_task = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_task.fit(betas, task_Y, runs)
    task_score_img = new_img_like(mean_fmri, searchlight_task.scores_)
    task_score_img.to_filename(func_path + "%s_task_logistic_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("task broke")
    pass

try:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression('l2', solver='liblinear')) 
    searchlight_task = nilearn.decoding.SearchLight(mask, process_mask_img=mask, estimator = pipeline, radius=6, n_jobs=24, verbose=0, cv=LeaveOneGroupOut())
    searchlight_task.fit(betas, feature_Y, runs)
    task_score_img = new_img_like(mean_fmri, searchlight_task.scores_)
    task_score_img.to_filename(func_path + "%s_feature_logistic_score.nii.gz" %subject)
    now = datetime.datetime.now()
    print(now)
except:
    print("feature broke")
    pass


### end