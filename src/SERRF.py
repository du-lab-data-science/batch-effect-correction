import numpy as np
from scipy.stats import rankdata
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRFRegressor as XGR
from sklearn.ensemble import RandomForestRegressor
import gc 
from sklearn.preprocessing import StandardScaler
import os 
import sys 
sys.path.append(os.path.abspath(".."))
from utils.utility_functions import TIC
"""
A python implementation of SERRF (Systematic Error Removal using Random Forest). 
The SERRF algorithm was originally implemented in R by Fan et al. in 2015

Parameters
---------
QC : pd.DataFrame, QC samples dataframe of shape (n_samples, n_signals)
Sample : pd.DataFrame, non-QC samples dataframe of shape (n_samples, n_signals)
Metadata : pd.DataFrame, Metadata information about QC and non-QC samples
n_batches : list, list of unique batch labels found in Metadata
signals : list, list of all signals (across batches)

Returns
-------
normed : pd.DataFrame
    - Normalized DataFrame (n_samples x n_signals)

References
---------
[1] Systematic Error Removal using Random Forest (SERRF) 
for Normalizing Large-Scale Untargeted Lipidomics Data 
Sili Fan, Tobias Kind, Tomas Cajka, Stanley L. Hazen, W. H. Wilson Tang, 
Rima Kaddurah-Daouk, Marguerite R. Irvin, Donna K. Arnett, 
Dinesh Kumar Barupal, and Oliver Fiehn Analytical Chemistry
DOI: 10.1021/acs.analchem.8b05592

"""
def compute_pool(QC, Sample, M, n_batches,signals):
    with Pool(processes=min(len(n_batches), 4)) as p:
        corr_arr = list(tqdm(p.imap(return_corrs_train_and_target, [(batch, QC, Sample, M) for batch in n_batches]),total=len(n_batches),desc='Computing Correlation Matrices'))
        args_for_intersection = [(corrs_train, corrs_target, signals) for corrs_train, corrs_target in corr_arr]
    
        intersection = list(tqdm(p.imap(return_corr_intersection,args_for_intersection),total=len(n_batches),desc='Computing Intersection'))
        args_for_sys_error_pred = [(signals,QC,Sample,batch,M,intersection[i]) for i,batch in enumerate(n_batches)]
    
        print("Computing Systematic Error...")
        sys_err_pred = list(p.imap(systematic_error_prediction,args_for_sys_error_pred))
    return sys_err_pred


def return_corrs_train_and_target(arg):
    batch,QC,Sample,M = arg
    train = QC.groupby(M['batch']).get_group(batch).to_numpy()
    target = Sample.groupby(M['batch']).get_group(batch).to_numpy()
    rank_train = np.apply_along_axis(rankdata,axis=0,arr=train)
    rank_target = np.apply_along_axis(rankdata,axis=0,arr=target)
    corrs_train = np.abs(np.corrcoef(rank_train,rowvar=False))
    corrs_target = np.abs(np.corrcoef(rank_target,rowvar=False))
    return (corrs_train,corrs_target)
def return_corr_intersection(results):
    corrs_train, corrs_target, signals = results
    intersection = {signal:None for signal in signals}
    for idx,signal in enumerate(signals):
        signal_train = pd.Series(corrs_train[:,idx],name=signal,index=signals)
        signal_target = pd.Series(corrs_target[:,idx],name=signal,index=signals)
        #sort values to get highest correlated peaks
        signal_train = signal_train.sort_values(ascending=False)
        signal_target = signal_target.sort_values(ascending=False)
        #remove self-correlation 
        signal_train.drop(signal_train.name,inplace=True)
        signal_target.drop(signal_target.name,inplace=True)
        intersection[signal] = intersect(signal_train,signal_target,n=10)
    return intersection
def intersect(x,y,n=10):
    inter = []
    j = 0
    x = x.index.to_list()
    y = y.index.to_list()
    for sig in x:
        if sig in y:
            inter.append(sig)
            if len(inter) == n:
                break
    return inter
def select_signals(signal,batch,QC,Sample,M,intersection,qc=True,):
    if qc:
        temp = QC.copy()
        temp['batch'] = M['batch']
        temp = temp[temp['batch'] == batch]
    else:
        temp = Sample.copy()
        temp['batch'] = M['batch']
        temp = temp[temp['batch'] == batch]
    x = temp[intersection[signal]]
    return StandardScaler().fit_transform(x) 

def systematic_error_prediction(args):
    signals,QC,Sample,batch,M,intersection = args
    all = pd.concat([QC,Sample])
    train = QC.groupby(M['batch']).get_group(batch).copy()
    target = Sample.groupby(M['batch']).get_group(batch).copy()
    correct_qc = train.copy()
    correct_sample = target.copy()
    pred_dict = {signal: None for signal in signals}
    for signal in tqdm(signals):
        
        rfr = RandomForestRegressor(n_estimators=500,min_samples_split=5,random_state=42)
        mean_ = train[signal].mean()
        train_scale_y = train[signal] - mean_
        train_scale_x = select_signals(signal,batch=batch,QC=train,Sample=target,intersection=intersection,M=M,qc=True)
        test_scale_x = select_signals(signal,batch=batch,QC=train,Sample=target,M=M,intersection=intersection,qc=False)
        rfr.fit(train_scale_x,train_scale_y)
        fitted_values = rfr.predict(train_scale_x)
        predictions = rfr.predict(test_scale_x)
        pred_dict[signal] = predictions
        del rfr, train_scale_x, test_scale_x, train_scale_y
        gc.collect()
        
        correct_qc.loc[:, signal] = correct_qc[signal] / (
            (fitted_values + correct_qc[signal].mean()) / all.loc[all.index.isin(QC.index), signal].mean())
     
        t2 = correct_sample[signal] / ((predictions + (correct_sample[signal].mean() - np.mean(predictions))) / Sample[signal].median())
        
        t2[t2 < 0] = correct_sample[signal][t2 < 0]
        
        correct_sample[signal] = t2
        
        correct_qc.loc[:, signal] = correct_qc[signal] / (correct_qc[signal].median() / all.loc[all.index.isin(QC.index), signal].median())
      
        correct_sample.loc[:,signal] = correct_sample[signal] / (correct_sample[signal].median() / all.loc[all.index.isin(Sample.index),signal].median())
  
        del fitted_values, predictions,t2
        gc.collect()
    current_batch = pd.concat([correct_qc,correct_sample])
    return (current_batch,pred_dict)
def find_outlier(sig,coef=3):
    Q3,Q1 = np.percentile(sig,[75,25])
    iqr = Q3-Q1
    lower_bound = Q1 - coef * iqr
    upper_bound = Q3 + coef * iqr

    return (sig < lower_bound) | (sig > upper_bound)
def adjust_correction(current_batch,pred_dict):
    current_batch_pc = current_batch.copy()
    correct_sample = current_batch[current_batch.index.isin(Sample.index)].copy()

    inf_bool = current_batch.apply(lambda x: np.isinf(x))
    n_infs = current_batch[inf_bool].sum().values.sum()
    outlier_bool = current_batch.apply(lambda x: find_outlier(x))
    if inf_bool.any().any():
        for signal in current_batch.columns:
            mask = inf_bool[signal]
            if mask.any():
                std_dev = current_batch.loc[~np.isinf(current_batch[signal]), signal].std(skipna=True) * 0.01
                current_batch.loc[mask, signal] = np.random.normal(0, std_dev, size=n_infs)
    if outlier_bool.any().any(): #check if any outliers and replace
        for signal in pred_dict:
            outlier_mask = outlier_bool[signal]
            if outlier_mask.any(): #check if any outliers in column before preceding
                predictions = pred_dict[signal]
                full_attempt = correct_sample[signal] - (predictions + correct_sample[signal].mean() - Sample[signal].median())
                attempt = full_attempt[outlier_mask]
                is_high_outlier = current_batch.loc[outlier_mask, signal].mean() > current_batch[signal].mean()
                if is_high_outlier:
                    if attempt.mean() < current_batch.loc[outlier_mask, signal].mean():
                        sample_mask = current_batch.index.isin(Sample.index)
                        combined_mask = sample_mask & outlier_mask
                        current_batch.loc[combined_mask, signal] = attempt.values
                else:
                     if attempt.mean() > current_batch.loc[outlier_mask, signal].mean():
                         sample_mask = current_batch.index.isin(Sample.index)
                         combined_mask = sample_mask & outlier_mask
                         current_batch.loc[combined_mask, signal] = attempt.values
    mask = current_batch.loc[current_batch.index.isin(Sample.index)] < 0
    current_batch = current_batch.mask(mask, current_batch_pc)
    normed = current_batch
    return normed 
        
    
if __name__ == "__main__":
    D = pd.read_csv("Data/2-peak_area_after_filling_missing_values.csv").drop(columns=['position','mz','rt'])
    
    M = pd.read_csv("Data/sample_metadata_all_batches.csv")
    M.set_index('sample_name',inplace=True)
    D.set_index("name",inplace=True)
    D = D.T #shape = (n_samples,n_signals)
    D = D[~D.index.str.contains("AOU_S_0104")]
    D = TIC(D,scale=True)
    #D = D.sample(5000,axis=1,random_state=42)
    signals = D.columns.to_list()
    n_batches  = M.batch.unique()
    for index,row in M.iterrows():
        if row.sample_type == 'sp':
            M.loc[index,"mask_sample_type"] = 'qc'
        elif row.sample_type == "blank":
            M.loc[index,'mask_sample_type'] = 'BLANK'
        else:
            M.loc[index,'mask_sample_type'] = 'sample'

    #sort data by sample_type
    grouping = D.groupby(M['mask_sample_type'])
    QC = grouping.get_group("qc")
    Sample = grouping.get_group('sample')
    all = pd.concat([QC,Sample])
    sys_err_pred = compute_pool(signals=signals,QC=QC,Sample=Sample,M=M,n_batches=n_batches)
    normed_dict = {}
    for batch,i in enumerate(sys_err_pred):
        current_batch = i[0]
        pred_dict = i[1]
        normed_dict[batch+1] = adjust_correction(pred_dict=pred_dict,current_batch=current_batch)
    print("Removing Systematic Error")
    normed = pd.concat(normed_dict[x] for x in normed_dict)
    A = normed.loc[normed.index.isin(Sample.index)].median()
    B = all.loc[all.index.isin(QC.index)].median()
    C = all.loc[all.index.isin(Sample.index)].median()
    D = all.loc[all.index.isin(Sample.index)].std()
    E = normed.loc[normed.index.isin(Sample.index)].std()
    F = normed.loc[~normed.index.isin(Sample.index)].median()
    c = (A + ((B - C) / D * E)) / F
    c = c.apply(lambda x: x if x>0 else 1)
    normed[normed.index.isin(QC.index)] = normed[normed.index.isin(QC.index)] * c
    print("Saving Data")
    normed.to_csv("serrf_imp.csv")
    print("Done")
    

