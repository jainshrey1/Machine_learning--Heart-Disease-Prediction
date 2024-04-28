"""
This file contains functions to train and evaluate models. The aim of main function is 
to train and evaluate models using different imputation techniques, balancing techniques and algorithms. 
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer,KNNImputer
from IPython.display import display 


from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE,KMeansSMOTE,RandomOverSampler,SVMSMOTE
from imblearn.under_sampling import ClusterCentroids,AllKNN,CondensedNearestNeighbour,EditedNearestNeighbours,InstanceHardnessThreshold,RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from smote_variants import MWMOTE
import re


import sys
sys.path.append("../")

from utils.metrics import get_performances


# imputers_num = {"SimpleImputer_mean":SimpleImputer()}
# imputers_cat = {"SimpleImputer_mean":SimpleImputer(strategy='most_frequent')}

imputers = [
    [
        ["SimpleImputer_mean",SimpleImputer()],
        ["SimpleImputer_mode",SimpleImputer(strategy='most_frequent')]
    ],
    [
        ["KNNImptuer",KNNImputer()],
        ["SimpleImputer_mode",SimpleImputer(strategy='most_frequent')]
    ]
        ]



balancers = [SMOTE,MWMOTE,ADASYN,None,AllKNN]

algorithms = {"LogisticRegression":LogisticRegression(),
          "DecisionTree":DecisionTreeClassifier(),
          "NaiveBayes":GaussianNB(),
          "KNN":KNeighborsClassifier(),
          "SVM":SVC(),
          "RandomForest":RandomForestClassifier(),
          "GradientBoosting":GradientBoostingClassifier(),
          "Bagging":BaggingClassifier(),
          "XGBoost":XGBClassifier()}





from copy import deepcopy

import pandas as pd


def train_models(path,target_var,path_to_save):
    
    
    """
    
    This function trains and evaluates models using different imputation techniques, balancing techniques and algorithms.
    The aim of this function is to compare the performance of different models using different techniques, and to find the best combination of techniques.
    This is not model optimization, but a way to have a high level view of the data and the models.
    
    
    Parameters:
    -----------
    path: str
        The path to the data.
    target_var: str
        The target variable.
    path_to_save: str
        The path to save the results.
        
        
    Returns:
    --------
    pd.DataFrame
        A dataframe with the results.
    """
    
    performance_df = {"Imputer":[],
                      "Imbalance":[],
                      "Algorithm":[],
                      "Train-F-1":[],
                      "Train-Accuracy":[],
                      "Train-Kappa":[],
                      "Train-TP":[],
                      "Train-FP":[],
                      "Train-TN":[],
                      "Train-FN":[],
                      "Train-Recall":[],
                      "Train-Precision":[],
                      "Train-AUC":[],
                      "Train-FalseNegativeRate":[],
                      "Train-FalsePositiveRate":[],
                      "Test-F-1":[],
                      "Test-Accuracy":[],
                      "Test-Kappa":[],
                      "Test-TP":[],
                      "Test-FP":[],
                      "Test-TN":[],
                      "Test-FN":[],
                      "Test-Recall":[],
                      "Test-Precision":[],
                      "Test-AUC":[],
                      "Test-FalseNegativeRate":[],
                      "Test-FalsePositiveRate":[],}
    
    
    df = deepcopy(pd.read_csv(path))
    
    
    try:
        df = df[[col for col in df.columns if not re.match('TIME',col) and col not in ['RANDID','DEATH']]]
    except:
        pass
    
    print(df.shape)
    
    features = list(df.columns)
    features.remove(target_var)
    df = df[features + [target_var]]
    
    counts = df[features].nunique().to_frame().reset_index().rename(columns={"index":"Feature",0:"N Uniques"})
    
    cat_varaibles = counts[counts['N Uniques'] < 5]['Feature'].values
    num_varaiables = counts[counts['N Uniques'] > 5]['Feature'].values
    
    

    
    
    # for i_name,imputer in zip(imputers_cat.items(),imputers_num.items()):
    
    for imputer in imputers:

        ddf  = df.copy()
        x_cat = df[cat_varaibles].copy()
        x_num = df[num_varaiables].copy()
        
        # print(imputer)
        # quit()
    
    
        num_imputer,cat_imputer = imputer[0][1],imputer[1][1]
        num_imputer_name,cat_imputer_name = imputer[0][0],imputer[1][0]
        
        # impute, the first version of imputing was using entire data, the split
        # this time we are going to split, then impute
        x_cat = cat_imputer.fit_transform(x_cat)
        x_num = num_imputer.fit_transform(x_num)
        
        # 
        ddf[cat_varaibles] = x_cat
        ddf[num_varaiables] = x_num
        
        X_full,y_full = ddf.iloc[:,:-1].values,ddf.iloc[:,-1].values
        
        # balance
        for balancer in balancers:
            
            
            try:
                b_name = balancer.__name__
            except:
                b_name = 'OriginalData'
            
            if b_name != 'OriginalData':
                
                balancer = balancer()
                
                X_balanced,y_balaanced = balancer.fit_resample(X_full,y_full)
                
            else:
                X_balanced,y_balaanced = deepcopy(X_full),deepcopy(y_full)
            
            # train test and scale
            
            X_train,X_test,y_train,y_test = train_test_split(X_balanced,y_balaanced,random_state=123,test_size=.2)
            
            scaler = StandardScaler()
            
            scaler = scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            
            # train
            for m_name,model in algorithms.items():
                
                        
                performance_df['Imputer'].append(num_imputer_name + "__"+cat_imputer_name)                
                performance_df['Imbalance'].append(b_name)
                performance_df['Algorithm'].append(m_name)
                
                model = model.fit(X_train,y_train)
                
                # train performance
                
                y_pred = model.predict(X_train)
                
                acc,f1,tn,fp,fn,tp,kappa_score,recall,precision,fpr,fnr,auc = get_performances(y_true=y_train,
                                                                                          y_pred=y_pred)
                
                
                performance_df['Train-Accuracy'].append(acc)
                performance_df['Train-F-1'].append(f1)
                performance_df['Train-Kappa'].append(kappa_score)
                performance_df['Train-Precision'].append(precision)
                performance_df['Train-Recall'].append(recall)
                performance_df['Train-AUC'].append(auc)
                performance_df['Train-TP'].append(tp) 
                performance_df['Train-TN'].append(tn)               
                performance_df['Train-FP'].append(fp)               
                performance_df['Train-FN'].append(fn)    
                performance_df['Train-FalsePositiveRate'].append(fpr)
                performance_df['Train-FalseNegativeRate'].append(fnr)           
              
                # test performance
                
                y_pred = model.predict(X_test)


                acc,f1,tn,fp,fn,tp,kappa_score,recall,precision,fpr,fnr,auc = get_performances(y_true=y_test,
                                                                                          y_pred=y_pred)
                
                performance_df['Test-Accuracy'].append(acc)
                performance_df['Test-F-1'].append(f1)
                performance_df['Test-Kappa'].append(kappa_score)
                performance_df['Test-Precision'].append(precision)
                performance_df['Test-Recall'].append(recall)
                performance_df['Test-AUC'].append(auc)
                performance_df['Test-TP'].append(tp) 
                performance_df['Test-TN'].append(tn)               
                performance_df['Test-FP'].append(fp)               
                performance_df['Test-FN'].append(fn)               
                performance_df['Test-FalsePositiveRate'].append(fpr)
                performance_df['Test-FalseNegativeRate'].append(fnr)   
                
                
     
    performance_df = pd.DataFrame(performance_df).melt(id_vars=['Algorithm','Imputer','Imbalance'],var_name='Metric',value_name='Score') 
    
        
    
    # for now, we are not going to use the confusion matrix values
    performance_df = performance_df[performance_df['Metric'].apply(lambda x: False if x.split("-")[1] in ['TP','TN','FP','FN','FalsePositiveRate','FalseNegativeRate'] else True)]
    
    performance_df['Score'] = round(performance_df['Score'],2)

    performance_df['Set'] = performance_df['Metric'].apply(lambda x: x.split("-")[0])
  
    metrics_map = {}

    for met in performance_df.Metric.unique():

        if met in ['Train-F-1','Test-F-1']:
            val = 'F-1'
        else:
            val = met.split("-")[1]

        metrics_map[met] = val
        
    performance_df['MainMetric'] = performance_df['Metric'].map(metrics_map)

    performance_df.to_csv(path_to_save,index=False)
    # performance_df.
    return performance_df
            
            
            
            