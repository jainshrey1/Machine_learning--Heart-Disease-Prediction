from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import numpy as np
from itertools import product
import pandas as pd


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.impute import KNNImputer,SimpleImputer
import json
import shutil
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics  import roc_auc_score

import sys
from utils.get_parameters import get_combinations


from joblib import dump

sys.path.append("../")
from utils.metrics import  get_performances
from utils.data_preparation import balance_impute_data

from utils.get_parameters import get_params

def find_best_model(algorithm,balancer,data_path=None,df=None,target='CVD',test_size=.2,imputer=None,ovewrite=False):
    
    
    
    
    # to avoid relative path erros, change working direcoty to main for this file
    if __name__ != "__main__":
        # print("Yes")
        try:
            os.chdir("./utils")
        except:
            pass
        
        
        
    algorithm_name = algorithm.__name__
    
    try:
        parameters = get_params(algorithm_name)
    except:
        print("Model not found!!")
        return None,None,None
        
   
    # if model traing ask
    dest_folder = os.path.join("../results/",algorithm_name)
    
    if os.path.exists(dest_folder):
        if len(os.listdir(dest_folder)) > 0 and not ovewrite:
        
            ask = input(f"\n{algorithm_name} hyperparameter tuning was already done. Do you want to do it again? [y/[other key]] ")
            
            if ask.lower() != 'y':
                return None,None,None
            
            shutil.rmtree(dest_folder)

    if df is None:
            
        df = pd.read_csv(data_path)
    
    # get the data ready for modeling
    
    X_train,X_test,y_train,y_test,cat_imputer_name,num_imputer_name,balancer_name = balance_impute_data(df=df,
                                                                                                        balancer=balancer,
                                                                                                        imputer=imputer,
                                                                                                        test_size=test_size,
                                                                                                        target=target)
    



    n_params = len(parameters)
    
    
    
    best_params = None
    best_auc = 0
        
    for i,param in enumerate(parameters):
        


        
        model = algorithm().set_params(**param)

        
        if i % 20 == 0:
            
            x_valid_train,x_valid_test,y_valid_train,y_valid_test = train_test_split(X_train,y_train,test_size=.2)


        try:
            model = model.fit(x_valid_train,y_valid_train)
        except Exception as e:
            
            # print(str(e))
            continue
        
        
        
        # test performance
        y_pred_test = model.predict(x_valid_test)
        test_auc = roc_auc_score(y_valid_test,y_pred_test)

        
        # train performance
        y_pred_train = model.predict(x_valid_train)
        train_auc = roc_auc_score(y_valid_train,y_pred_train)

        if best_auc < test_auc:
            
            best_auc = test_auc
            best_params = model.get_params()
        
        
        print(f"{i+1}/{n_params} {algorithm_name}:\n\t[Train AUC: {train_auc}, Test AUC: {test_auc}]\n\t")    
            
    
        # break
    # best model
    
    best_model = algorithm().set_params(**best_params)
    
    model = best_model.fit(X_train,y_train)
    
    # test performance
    test_prediction = model.predict(X_test)
    
    output_test = get_performances(y_pred=test_prediction,y_true=y_test,return_dict=True)
    output_test = pd.DataFrame(output_test).T.reset_index().rename(columns={"index":"Metric",0:"Score"})
    output_test['Set'] = 'Test'
    
    # train performance
    train_prediction = model.predict(X_train)
    
    output_train = get_performances(y_pred=train_prediction,y_true=y_train,return_dict=True)
    output_train = pd.DataFrame(output_train).T.reset_index().rename(columns={"index":"Metric",0:"Score"})
    output_train['Set'] = 'Train'
    
    
    output = pd.concat([output_train,output_test])
    

    output['Algorithm'] = algorithm_name
    output['ImputerCat'] = cat_imputer_name
    output['ImputerNum'] = num_imputer_name
    output['Imbalance'] = balancer_name    
    
    
    output = output[['Algorithm','Imbalance','ImputerCat','ImputerNum','Set','Metric','Score']]


    # save the best model
    model_path = os.path.join("../models")
    if not os.path.exists(model_path):
        
        os.makedirs(model_path)
        
    dump(model,os.path.join(model_path,algorithm_name+"_best_model.pkl"))

    # perf_file_name = "log_regression_best_model_output.csv" if perf_file_name is None else perf_file_name
    # param_file_name = "log_regression_best_model_param.json" if param_file_name is None else param_file_name
    # param_file_name = os.path.join("../results/",algorithm_name)
    
    if not os.path.exists(dest_folder):
        
        os.makedirs(dest_folder)
    # if not os.path.exists(param_file_name):
        
    #     os.makedirs(param_file_name)
        
    output.to_csv(os.path.join(dest_folder,algorithm_name+"_best_model_perfomance.csv"),index=False)

    
    output.to_csv(os.path.join(dest_folder,algorithm_name+"_best_model_perfomance.csv"),index=False)
    
    with open(os.path.join(dest_folder,algorithm_name+"_best_model_params.json"), 'w') as file:
        
        json.dump(best_params,file)
        
        
    print(f"{algorithm_name} hyperparameter tuning was done!!")
    
    return best_model,best_params,output


def train_best_models(df=None,data_path=None):
    
    df = pd.read_csv(data_path) if df is None else df   
    
    #  get combinations of model,balance,imputer
    combinations = get_combinations()
    
    # print(combinations)
    # quit()
    
    
    best_results_for_all_models = None
  
    for combination in combinations:

        algorithm, imputer,balanc = combination
        # print(combination)
        # quit()
        print(algorithm.__name__,balanc.__name__,imputer)

        best_model,best_params,output = find_best_model(algorithm=algorithm,
                                                    balancer=balanc,
                                                    imputer=imputer,
                                                    df=df)

        if best_results_for_all_models is None:
            best_results_for_all_models = output
            continue
        
        best_results_for_all_models = pd.concat([best_results_for_all_models,output])
        
    best_results_for_all_models.to_csv("../results/general/best_results_all_models.csv",index=False)
        
    
        
if __name__ == "__main__":
    
    
    
    find_best_model(SVC,data_path="../../data/initial_data/frmgham2_project_data.csv",
                    balancer=SMOTE,test_size=.2,SimpleImputer_mean = SimpleImputer(),SimpleImputer_mode = SimpleImputer(strategy='most_frequent'))
    
    
    