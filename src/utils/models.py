from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from itertools import product
import pandas as pd


from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import json
import shutil
import os

from sklearn.metrics  import cohen_kappa_score
from metrics import  get_performances



def logistic_regression(path,balancer,path_to_save,perf_file_name=None,param_file_name=None,target='CVD',test_size=.2,**imputer):
    
    
    
    """
    Pipeline with GridSearch to find the best LogRegression model
    """
    df = pd.read_csv(path)
    
    try:
        df = df[[col for col in df.columns if not re.match('TIME',col) and col not in ['RANDID','DEATH']]]
    except:
        pass
    features = list(df.columns)
    features.remove(target)
    counts = df[features].nunique().to_frame().reset_index().rename(columns={"index":"Feature",0:"N Uniques"})
    
    cat_varaibles = counts[counts['N Uniques'] < 5]['Feature'].values
    num_varaiables = counts[counts['N Uniques'] > 5]['Feature'].values
    
    # prepare data
    
    ddf  = df.copy()
    x_cat = df[cat_varaibles].copy()
    x_num = df[num_varaiables].copy()
        

    # impute missings
    
    imputer = list(imputer.items())
    num_imputer_name,num_imputer = imputer[0]
    cat_imputer_name,cat_imputer = imputer[1]    
    # impute, the first version of imputing was using entire data, the split
    # this time we are going to split, then impute
    x_cat = cat_imputer.fit_transform(x_cat)
    x_num = num_imputer.fit_transform(x_num)
    
    # 
    ddf[cat_varaibles] = x_cat
    ddf[num_varaiables] = x_num
    
    X,y = ddf[features].values,ddf[target].values
        
        
    # balance
    
    balancer_name = balancer.__name__
    balancer = balancer()
    X,y = balancer.fit_resample(X,y)
    
    
    # split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123,test_size=test_size)

    # scale
    scaler = StandardScaler()
    
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    
    penalties = ['l1', 'l2', 'elasticnet', None]
    # duals = [True,False]
    Cs =  [0.01, 0.1, .3, .5, 1]
    # fit_intercept = True,False
    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    max_iters = [100,1000,10000]
    l1_ratios = np.linspace(0.1,1,5,endpoint=True)
    
    
    parameters = list(product(penalties,Cs,solvers,max_iters,l1_ratios))
    
    n_params = len(parameters)
    
    
    best_params = None
    best_kappa = 0
        
    for i,param in enumerate(parameters):
        
        penalty,c,solver,max_iter,l1_ratio = param
        
        model = LogisticRegression(penalty=penalty,C=c,solver=solver,max_iter=max_iter,l1_ratio=l1_ratio)
        
        try:
            model = model.fit(X_train,y_train)
        except:
            continue
        
        
        
        y_pred_test = model.predict(X_test)
        
        # acc,f1,tn,fp,fn,tp,kappa_score,recall,precision,fpr,fnr = get_performances(y_pred=y_pred_test,y_true=y_test)
        
        
        y_pred_train = model.predict(X_train)
        
        # scores
        kappa_test = cohen_kappa_score(y_test,y_pred_test)
        kappa_train = cohen_kappa_score(y_train,y_pred_train)
        
        if best_kappa < kappa_test:
            
            best_kappa = kappa_test
            best_params = model.get_params()
        
        print(f"{'-'*20} {i+1}/{n_params}[Train Kappa: {kappa_train}, Test Kappa: {kappa_test}]{'-'*20}")
    
    # best model
    
    best_model = LogisticRegression().set_params(**best_params)
    
    model = model.fit(X_train,y_train)
    
    # test performance
    test_prediction = model.predict(X_test)
    
    output_test = get_performances(y_pred=test_prediction,y_true=y_test,return_dict=True)
    
    output_test = pd.DataFrame(output_test).T.reset_index().rename(columns={"index":"Metric",0:"Score"})
    output_test['Set'] = 'Test'
    
    # train performance
    train_prediction = model.predict(X_train)
    
    output_train = get_performances(y_pred=train_prediction,y_true=y_train,return_dict=True)
    output_train = pd.DataFrame(output_test).T.reset_index().rename(columns={"index":"Metric",0:"Score"})
    output_train['Set'] = 'Train'
    
    
    output = pd.concat([output_train,output_test])
    output['Algorithm'] = 'LogisticRegression'
    output['ImputerCat'] = cat_imputer_name
    output['ImputerNum'] = num_imputer_name
    output['Imbalance'] = balancer    
    



    if not os.path.exists(path_to_save):
        
        os.makedirs(path_to_save)
        
    else:
        
        answer = input("Path exists, do you want ovewrite it (y/...)")
        
        if answer == "y":
            
            shutil.rmtree(path_to_save)
            os.makedirs(path_to_save)
            
    
    perf_file_name = "log_regression_best_model_output.csv" if perf_file_name is None else perf_file_name
    param_file_name = "log_regression_best_model_param.json" if param_file_name is None else param_file_name

    output.to_csv(os.path.join(path_to_save,perf_file_name),index=False)
    
    with open(os.path.join(path_to_save,param_file_name), 'w') as file:
        
        json.dump(best_params,file)
        
        
    print("Logist regression hyperparametere tuning was done!!")


    
if __name__ == "__main__":
    
    
    
    path_to_save = "../results/log_reg/"
    

    
    logistic_regression(path="../../data/initial_data/frmgham2_project_data.csv",
                        path_to_save=path_to_save,balancer=SMOTE,test_size=.2,KNNImputer = KNNImputer(),SimpleImputer = SimpleImputer(strategy='most_frequent'))
    
    
    
    