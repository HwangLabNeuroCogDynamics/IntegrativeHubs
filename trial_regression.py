import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.formula.api as smf
import scipy.linalg as linalg
from joblib import Parallel, delayed

################################
## Fit RSA model to similarity matrices. use Argon
################################
included_subjects = input()

if __name__ == "__main__":

    now = datetime.now()
    print("Start time: ", now)
    permute = False

    #### setup
    ## relevant paths
    data_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/"
    out_dir = "/Shared/lss_kahwang_hpc/data/ThalHi/RSA/trialwiseRSA/stats/"
    coef_fn = 'whole_brain'
    # model_syntax = ["coef ~ 1 + context + task + response + color + shape + stim + identity + EDS + IDS + condition + error" +
    #                 "+ context:EDS + context:IDS + task:EDS + task:IDS + feature:EDS + feature:IDS" + 
    #                 "+ response:EDS + response:IDS" + 
    #                 "+ context:error + feature:error + task:error + response:error" +
    #                 "+ context:color + context:shape "
    #                 ]
    
    #model files to load
    models = ["context", "color", "shape", "task", "response", "stim", "EDS", "IDS", "Stay", "condition", "feature", "feature_color", "feature_shape", "error", "identity", "rt", "response_repeat"]
    
    #simple model
    #model_syntax = ["coef ~ 1 + context + task + response + color + shape + stim + feature + error"]
    #model_syntax = ["coef ~ 1 + context + task  + feature_color + feature_shape + response_repeat+ identity:Stay"]
        ## is there a way to select "context relevant features?"

    #task switch model

    model_syntax = ["coef ~ 1 + context + task + response + feature_color + feature_shape + stim + error + " +
                    "error*EDS*context + error*EDS*feature_color + error*EDS*feature_shape + error*IDS*context + error*IDS*feature_color + error*IDS*feature_shape + error*Stay*context + error*Stay*feature_color + error*Stay*feature_shape"]
    
    #model_syntax = ["coef ~ 0 + context + task + response + feature_color + feature_shape + stim + error + " +
    #                "error*EDS*context + error*EDS*feature_color + error*EDS*feature_shape + error*IDS*context + error*IDS*feature_color + error*IDS*feature_shape + error*Stay*context + error*Stay*feature_color + error*Stay*feature_shape"]
    
    # model_syntax = ["coef ~ 0 + context + task + response + feature_color + feature_shape + stim + Stay*identity +" +
    #                 "rt*EDS*context + rt*EDS*feature_color + rt*EDS*feature_shape + rt*IDS*context + rt*IDS*feature_color + rt*IDS*feature_shape + rt*Stay*context + rt*Stay*feature_color + rt*Stay*feature_shape + " +
    #                 "error*EDS*context + error*EDS*feature_color + error*EDS*feature_shape + error*IDS*context + error*IDS*feature_color + error*IDS*feature_shape + error*Stay*context + error*Stay*feature_color + error*Stay*feature_shape"]
    
    # model_syntax = ["coef ~ 0 + context + task + response + feature_color + feature_shape + stim + Stay*identity + error +" +
    #                  " EDS*context + EDS*feature_color + EDS*feature_shape + IDS*context + IDS*feature_color + IDS*feature_shape + Stay*context + Stay*feature_color + Stay*feature_shape"
    #                  ]
    num_permutations = 4096

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
            df['coef'] = coef[0].flatten()[lower_triangle_usable_inds]
            for m in models:
                df[m] = regressors[m].flatten()[lower_triangle_usable_inds] - np.nanmean(regressors[m].flatten()[lower_triangle_usable_inds]) 
            
            # create interactions
            #df["context:color"] = df["context"] * df["color"]
            #df["context:shape"] = df["context"] * df["shape"]
            #df["context:color:shape"] = df["context"] * df["shape"] *df["color"]
            # df["context:EDS"] = df["context"] * df["EDS"]
            # df["color:IDS"] = df["color"] * df["IDS"]
            # df["shape:IDS"] = df["shape"] * df["IDS"]
            #X = df[['context', 'task', 'response', 'color', 'shape',  "stim", 'EDS', 'IDS', 'condition', 'context:EDS', 'color:IDS', 'shape:IDS']]
            #X = np.column_stack((np.ones(len(X)), X))
            #df = df.dropna()
            model = smf.ols(formula = model_syntax[0], data=df)
            X = model.exog
            permuted_results = np.zeros((num_ROIs, X.shape[1],num_permutations)) 

            for p in np.arange(num_permutations):

                print("now permutation number: ", p+1)
                Y = model.endog.reshape(coef.shape[0],-1)[lower_triangle_usable_inds]
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


            ## this is the parallel version but mysteriorsly it is sig slower
            # Y = coef.reshape(coef.shape[0], -1)[:, lower_triangle_usable_inds]
            # #Y = model.endog
            # Y = Y.T
            # Y[np.isnan(Y)] = np.nanmean(Y)

            # def permute_and_fit(p):
            #     print("now permutation number: ", p + 1)

            #     # permute X
            #     np.random.seed(p)
            #     random_inds = np.random.permutation(X.shape[0])
            #     permuted_X = X[random_inds, :]

            #     # run regression, use scipy to fit to all ROIs at once
            #     try:
            #         result, _, _, _ = linalg.lstsq(Y, permuted_X)
            #     except:
            #         result = np.nan * np.ones((num_ROIs, X.shape[1]))
            #     return result

            # results = Parallel(n_jobs=4)(delayed(permute_and_fit)(p) for p in np.arange(100))

            # for p in np.arange(num_permutations):
            #     permuted_results[:, :, p] = results[p]
            
            # # #save output
            # np.save(out_dir + "%s_%s_permutated_stats.npy" %(s, coef_fn), permuted_results)

        #else:
        results = []
        for r in np.arange(num_ROIs):
            print("now running ROI number: ", r+1)
            # build dataframe
            df = pd.DataFrame()
            df['coef'] = coef[r].flatten()[lower_triangle_usable_inds]
            for m in models:
                df[m] = regressors[m].flatten()[lower_triangle_usable_inds] - np.nanmean(regressors[m].flatten()[lower_triangle_usable_inds]) 
            df = df.dropna()
            #run regression
            model = smf.ols(formula = model_syntax[0], data=df).fit()
            
            #print(model.summary())
            #from statsmodels.stats.outliers_influence import variance_inflation_factor
            #variance_inflation_factor(df, 0)
             
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
                    