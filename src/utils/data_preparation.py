

"""
This file contains functions to prepare data for modeling.

"""


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

def balance_impute_data(balancer,imputer,data_path=None,df=None,test_size=.2,target='CVD'):
    
    
    """
    This function prepares data for modeling. 
    It reads the data, imputes the missing values, balances the data, scales the data, and splits the data into train and test sets.


    Parameters:
    -----------
    data_path: str
        The path to the data.  
    balancer: class
        The balancing technique. This should be a class from imblearn.over_sampling or imblearn.under_sampling.
    imputer: dict
        The imputation techniques. This should be a dictionary with two keys: one for numerical imputation and one for categorical imputation.
        The imputer value should be a class from sklearn.impute and already instantiated.
    test_size: float
        The size of the test set. The default value is .2
    target: str
        The target variable. The default value is 'CVD'
        
    Returns:
    --------
    tuple
        A tuple with the train and test sets and the names of the imputers and balancer.  
    """
    
    
    # read the dataset
    if df is  None:
        df = deepcopy(pd.read_csv(data_path))
    
    # if there are time columns, death, and RANDID, remove them
    try:
        df = df[[col for col in df.columns if not re.match('TIME',col) and col not in ['RANDID','DEATH']]]
    except:
        pass
    
    # get the features
    features = list(df.columns)
    
    # print(features)
    # quit()
    features.remove(target)
    
    # get counts of each feature unique value, if the unique values are less than 5, they are considered as categorical (this works only for this dataset, it is not a general rule)
    counts = df[features].nunique().to_frame().reset_index().rename(columns={"index":"Feature",0:"N Uniques"})
    
    cat_varaibles = counts[counts['N Uniques'] < 5]['Feature'].values
    num_varaiables = counts[counts['N Uniques'] > 5]['Feature'].values
    
    # prepare data
    
    ddf  = df.copy()
    x_cat = df[cat_varaibles].copy()
    x_num = df[num_varaiables].copy()
        

    # impute missings
    
    # imputer = list(imputer.items())
    num_imputer_name,num_imputer = imputer[0],imputer[1]
    cat_imputer_name,cat_imputer = imputer[2],imputer[3]    
    # impute, the first version of imputing was using entire data, the split
    # this time we are going to split, then impute
    x_cat = cat_imputer.fit_transform(x_cat)
    x_num = num_imputer.fit_transform(x_num)
    
    # 
    ddf[cat_varaibles] = x_cat
    ddf[num_varaiables] = x_num
    
    X,y = ddf[features].values,ddf[target].values
        
        
    # balance
    
    try:
        balancer_name = balancer.__name__
        balancer = balancer()
        X,y = balancer.fit_resample(X,y)
    except:
        balancer_name = 'OriginalData'
    
    
    # split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123,test_size=test_size)

    # scale
    scaler = StandardScaler()
    
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    # if save:
        
    #     try:
    
    return X_train,X_test,y_train,y_test,cat_imputer_name,num_imputer_name,balancer_name


    
    