from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from itertools import product
import pandas as pd


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.impute import KNNImputer,SimpleImputer
import json
import shutil
import os

from sklearn.metrics  import cohen_kappa_score

import sys

from joblib import dump

sys.path.append("../")
from utils.metrics import  get_performances
from utils.data_preparation import balance_impute_data

def get_params(algoritm=None):
    
    
    """
    This functions is a container for all hyperparameters combinations for the algorithms.
    The only parameter is the algorithm name. 
    
    Parameters:
    -----------
    algorithm: str
        The name of the algorithm. The default value is None. The name should be the same as the algorithm name in sklearn.
        The opitions are:
            - LogisticRegression
            - DecisionTreeClassifier
            - SVC
            - KNeighborsClassifier
    Returns:
    --------
    list of dicts
        A list of dictionaries with hyperparameters combinates for the algorithm (list of dicts). 
    """
    
    
    
    
    # this lambda functions create a list of dictionaries for hyperparameters combinations
    get_param_dict = lambda names,params: [{names[j]:params_comb[j] for j in range(len(names))} for params_comb in params]
    
    match algoritm:
        case "LogisticRegression":
            
            # logistic regression
            log_penalties = ['l1', 'l2', 'elasticnet', None]
            log_Cs =  [0.01, 0.1, .3, .5, 1]
            log_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            log_max_iters = [100,1000,10000]
            log_l1_ratios = np.linspace(0.1,1,5,endpoint=True)



            log_params = list(product(log_penalties,log_Cs,log_solvers,log_max_iters,log_l1_ratios))
            log_params_names = ["penalty","C","solver","max_iter","l1_ratio"]

            log_params_dict = get_param_dict(log_params_names,log_params)
            
            return log_params_dict
        
        case "DecisionTreeClassifier":
            
            # decision tree
            tree_criterion = ["gini", "entropy", "log_loss"]
            tree_splitter = ["best", "random"]
            tree_max_depth = [2,4,6,8]
            tree_min_samples_split = [11,21,31]
            tree_min_samples_leaf = [3,5,7]
            tree_max_features = [10,15,"sqrt", "log2"]
            tree_random_state = [123,None]

            tree_params = list(product(tree_criterion,tree_splitter,tree_max_depth,tree_min_samples_split,
                                            tree_min_samples_leaf,tree_max_features,tree_random_state))


            tree_params_names = ["criterion","splitter","max_depth","min_samples_split","min_samples_leaf","max_features","random_state"]


            tree_params_dict = get_param_dict(tree_params_names,tree_params)
            
            return tree_params_dict
        
        case "SVC":
            
    
            # svm
            svm_C = [.3,.5,1.,2.]
            svm_kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
            svm_degree = [2,3,4]
            svm_gamma = ['scale', 'auto']
            svm_coef0 = [0,.1,.3,.5]
            svm_decision_function_shape = ['ovo', 'ovr']
            svm_random_state = [123,None]
            
            svm_params = list(product(svm_C,svm_kernel,svm_degree,svm_gamma,svm_coef0,svm_decision_function_shape,svm_random_state))
            svm_params_names = ["C","kernel","degree","gamma","coef0","decision_function_shape","random_state"]
            
            svm_params_dict = get_param_dict(svm_params_names,svm_params)
            
            return svm_params_dict
        
        case "KNeighborsClassifier":
            
            # knn
            
            knn_n_neighbors = [5,7,10]
            knn_weights = ['uniform', 'distance']
            knn_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
            knn_p = [1,2,3]
            
            knn_params =list(product(knn_n_neighbors,knn_weights,knn_algorithm,knn_p))
            knn_params_names = ['n_neighbors','weights','algorithm','p']
            
            knn_params_dict = get_param_dict(knn_params_names,knn_params)

            
            return knn_params_dict


        case _:
            
            raise ValueError("The algorithm name is not valid. \nPlease provide a valid algorithm name from the following list: \n\t['LogisticRegression', 'DecisionTreeClassifier', 'SVC', 'KNeighborsClassifier']")

def find_best_model(algorithm,data_path,balancer,target='CVD',test_size=.2,perf_file_name=None,param_file_name=None,imputer=None):
    
    
    
    
    # to avoid relative path erros, change working direcoty to main for this file
    if __name__ != "__main__":
        # print("Yes")
        os.chdir("./utils")
        

   
    
    # get the data ready for modeling
    
    X_train,X_test,y_train,y_test,cat_imputer_name,num_imputer_name,balancer_name = balance_impute_data(data_path = data_path,
                                                                                                        balancer=balancer,
                                                                                                        imputer=imputer,
                                                                                                        test_size=test_size,
                                                                                                        target=target)
    
    algorithm_name = algorithm.__name__
    parameters = get_params(algorithm_name)


    n_params = len(parameters)
    
    
    best_params = None
    best_kappa = 0
        
    for i,param in enumerate(parameters):
        

        
        model = algorithm().set_params(**param)

        
        
        try:
            model = model.fit(X_train,y_train)
        except:
            continue
        
        
        
        y_pred_test = model.predict(X_test)
            
        
        y_pred_train = model.predict(X_train)
        
        # scores
        kappa_test = cohen_kappa_score(y_test,y_pred_test)
        kappa_train = cohen_kappa_score(y_train,y_pred_train)
        
        if best_kappa < kappa_test:
            
            best_kappa = kappa_test
            best_params = model.get_params()
        
        print(f"{'-'*20} {i+1}/{n_params} {algorithm_name}:[Train Kappa: {kappa_train}, Test Kappa: {kappa_test}]{'-'*20}")
        
    
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
        
    dump(model,os.path.join(model_path,algorithm_name+"best_model.pkl"))

    # perf_file_name = "log_regression_best_model_output.csv" if perf_file_name is None else perf_file_name
    # param_file_name = "log_regression_best_model_param.json" if param_file_name is None else param_file_name
    perf_file_name = os.path.join("../results/",algorithm_name)
    param_file_name = os.path.join("../results/",algorithm_name)
    
    if not os.path.exists(perf_file_name):
        
        os.makedirs(perf_file_name)
    if not os.path.exists(param_file_name):
        
        os.makedirs(param_file_name)
        
    output.to_csv(os.path.join(perf_file_name,algorithm_name+"best_model_perfomance.csv"),index=False)

    
    output.to_csv(os.path.join(perf_file_name,algorithm_name+"best_model_perfomance.csv"),index=False)
    
    with open(os.path.join(param_file_name,algorithm_name+"best_model_params.json"), 'w') as file:
        
        json.dump(best_params,file)
        
        
    print(f"{algorithm_name} hyperparameter tuning was done!!")
    
    return best_model,best_params,output



        
if __name__ == "__main__":
    
    
    
    find_best_model(SVC,data_path="../../data/initial_data/frmgham2_project_data.csv",
                    balancer=SMOTE,test_size=.2,SimpleImputer_mean = SimpleImputer(),SimpleImputer_mode = SimpleImputer(strategy='most_frequent'))
    
    
    