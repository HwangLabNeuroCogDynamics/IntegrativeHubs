import numpy as np
import pandas as pd
import nibabel as nib
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.linalg as linalg

################################
## Fit RSA model to similarity matrices. use Argon
################################
included_subjects = input()
permute = False

if __name__ == "__main__":

    now = datetime.now()
    print("Start time: ", now)
    
    #### setup
    ## relevant paths
    data_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/"
    out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/"
    coef_fn = 'whole_brain'
    
    for s in [included_subjects]:
        print("now running subject: ", s)

        # load similarity matrix (coef) and RSA models
        coef = np.load(data_dir + "coefs/%s_" %s + coef_fn + "_coef.npy")

        num_trials = coef.shape[2]
        trial_per_run = 51
        num_runs = int(num_trials / trial_per_run)
        
        # Steph's code on removing within run trials
        tmp_mat = np.ones((num_trials,num_trials))
        for r in range(num_runs):
            for ii in range(int(num_trials/num_runs)):
                for jj in range(int(num_trials/num_runs)):
                    tmp_mat[ii+(int(num_trials/num_runs)*r),jj+(int(num_trials/num_runs)*r)] = -1
        tmp_vec = np.tril( tmp_mat, k=-1).flatten() # k=-1 should reduce to only entries below the diagonal
        lower_triangle_usable_inds = np.where(tmp_vec > 0)[0]
        print("number of usable cells from the trial by trial matrix: ", len(lower_triangle_usable_inds))

        # load models
        models = ["context", "color", "shape", "task", "response", "stim"]
        regressors = {}
        for m in models:
            regressors[m] = np.load(data_dir + "models/" + "%s_%s_model.npy" %(s, m))

        num_ROIs = coef.shape[0]
        
        # if running permutation
        if permute:

            # permute_results = {}
            # for m in models:
            #     permute_results[m] = np.zeros((num_ROIs, 1000))
            df = pd.DataFrame()
            for m in models:
                df[m] = regressors[m].flatten()[lower_triangle_usable_inds] - regressors[m].flatten()[lower_triangle_usable_inds].mean() 
            
            # create interactions
            df["context:color"] = df["context"] * df["color"]
            df["context:shape"] = df["context"] * df["shape"]
            df["context:color:shape"] = df["context"] * df["shape"] *df["color"]

            X = df[['context', 'task', 'response', 'color', 'shape',  "stim", 'context:color', 'context:shape', 'context:color:shape']]
            X = np.column_stack((np.ones(len(X)), X))
            permuted_results = np.zeros((num_ROIs, 10,1000)) 

            for p in np.arange(1000):

                print("now permutation number: ", p+1)
                Y = coef.reshape(coef.shape[0],-1)[:,lower_triangle_usable_inds]
                Y = Y.T
                Y[np.isnan(Y)] = np.nanmean(Y)
                
                #permute X
                np.random.seed(p)
                random_inds = np.random.permutation(X.shape[0])
                permuted_X =X[random_inds, :]
                
                #run regression, use scipy to fit to all ROIs at once
                try:
                    permuted_results[:,:,p], _, _, _ = linalg.lstsq(Y, permuted_X)
                except:
                    permuted_results[:,:,p] = np.nan
           
            #save output
            np.save(out_dir + "%s_%s_permutated_stats.npy" %(s, coef_fn), permuted_results)

        #else:
        results = []
        for r in np.arange(num_ROIs):
            print("now running ROI number: ", r+1)
            # build dataframe
            df = pd.DataFrame()
            df['coef'] = coef[r].flatten()[lower_triangle_usable_inds]
            for m in models:
                df[m] = regressors[m].flatten()[lower_triangle_usable_inds] - regressors[m].flatten()[lower_triangle_usable_inds].mean() 

            #run regression
            model = smf.ols(formula="coef ~ 1 + context + task + response + color + shape + stim + context:color + context:shape + context:color:shape", data=df).fit()
            #print(model.summary())
            tdf = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0].rename(columns={'P>|t|': 'pvalue'}).rename(columns={'index': 'parameter'})
            tdf['sub'] = s
            tdf['ROI'] = r +1
            tdf = tdf.reset_index().rename(columns={'index': 'parameter'})
            results.append(tdf)
        results_df = pd.concat(results, ignore_index=True)    
        results_df.to_csv(out_dir + "%s_%s_stats.csv" %(s, coef_fn))       

        # note, I have compared sm.ols and linalg.lstsq, they gave the same results


    now = datetime.now()
    print("End time: ", now)

#end
                    