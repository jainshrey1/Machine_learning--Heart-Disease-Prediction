"""
This file is for the model optimization. 

The main function is find_best_model which finds the best hyperparameters for a given algorithm, imputer, and balancer.
The function train_best_models trains the best models for all the combinations of algorithms, imputers, and balancers.
"""


import pandas as pd
import os
import shutil
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics  import roc_auc_score

from joblib import dump

import sys
sys.path.append("../")

from utils.get_parameters import get_combinations,get_params
from utils.metrics import  get_performances
from utils.data_preparation import balance_impute_data




def find_best_model(algorithm,balancer,data_path=None,df=None,target='CVD',test_size=.2,imputer=None,ovewrite=False):
    
    
    """
    The function finds the best hyperparameters for a given algorithm, imputer, and balancer. 
    This is a type of Grid Search from scratch.
    The steps are:
    
        - get algorithm hyperparameters from get_params by algorithm name
        - read the data
        - prepare the data for modeling using balance_impute_data function
        - iterate over the hyperparameters and fit the model
        - split train data into train and validation sets, resplit every 20 iterations
        - calculate the AUC for the validation and train sets; use the validation AUC to track the best model
        - finding the best model, fit the model on the train set and calculate the performance metrics on the train and test sets
        - save 
            - the best model in .pkl format
            - the best hyperparameters in .json format
            - the performance metrics in .csv format
            
        return the best model, best hyperparameters, and the performance metrics.
        
        
    Parameters:
    -----------
    algorithm: class
        The algorithm to optimize. This should be a class from sklearn or xgboost. 
        
    balancer: class
        The balancing technique. This should be a class from imblearn.over_sampling, imblearn.under_sampling or smote_variants 
        
    data_path: str
        The path to the data. The default value is None.
        
    df: pd.DataFrame
        The data. The default value is None.
        if both data_path and df are None, the function will raise an error.
        
    target: str
        The target variable. The default value is 'CVD'
        
    test_size: float
        The size of the test set. The default value is .2
        
    imputer: list
        The imputation techniques. This should be a list with four elements:
            the elements at index 0 and 2 are respectively the numerical and categorical imputers names in string format
            the elements at index 1 and 3 are respectively the numerical and categorical imputers classes instances from sklearn.impute
            
    ovewrite: bool
        If True, the function will overwrite the results if they already exist. The default value is False.
        If False, the function will ask the user if they want to overwrite the results if they already exist.
        
    Returns:
    --------
    tuple
        A tuple with the best model, best hyperparameters, and the performance metrics.
        (best_model,best_params,output)
        
    """
    
    # to avoid relative path erros, change working direcoty to main for this file
    if __name__ != "__main__":
        # print("Yes")
        try:
            os.chdir("./utils")
        except:
            pass
        
        
        
    # get the algorithm name by calling __name__ magic method
    algorithm_name = algorithm.__name__
    
    try:
        parameters = get_params(algorithm_name)
    except:
        print("Model not found!!")
        return None,None,None
        
   
    # define destination folder, if it already exists, ask the user if they want to overwrite the results
    # if ovewrite is True, delete the folder and its content
    dest_folder = os.path.join("../results/",algorithm_name)
    
    if os.path.exists(dest_folder):
        if len(os.listdir(dest_folder)) > 0 and not ovewrite:
        
            ask = input(f"\n{algorithm_name} hyperparameter tuning was already done. Do you want to do it again? [y/[other key]] ")
            
            if ask.lower() != 'y':
                return None,None,None
            
            shutil.rmtree(dest_folder)

    # read the data if df is None
    if df is None:
            
        df = pd.read_csv(data_path)
    
    
    # get the data ready for modeling
    X_train,X_test,y_train,y_test,cat_imputer_name,num_imputer_name,balancer_name = balance_impute_data(df=df,
                                                                                                        balancer=balancer,
                                                                                                        imputer=imputer,
                                                                                                        test_size=test_size,
                                                                                                        target=target)
    

    # number of hyperparameters
    n_params = len(parameters)
    
    
    # best model and best hyperparameters
    best_params = None
    best_auc = 0
        
    # iterate over the hyperparameters
    for i,param in enumerate(parameters):
        
        
        # create the model setting the hyperparameters        
        model = algorithm().set_params(**param)


        # for every 20 iterations, resplit the data into train and validation sets, this is a type of cross-validation        
        if i % 20 == 0:
            
            x_valid_train,x_valid_test,y_valid_train,y_valid_test = train_test_split(X_train,y_train,test_size=.2)
            
            
        # try to fit the model, if there is an error, skip the hyperparameters and continue with the next ones
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


        # compare the AUC of the validation set with the best AUC
        if best_auc < test_auc:
            
            best_auc = test_auc
            best_params = model.get_params()
        
        
        print(f"{i+1}/{n_params} {algorithm_name}:\n\t[Train AUC: {train_auc}, Test AUC: {test_auc}]\n\t")    
            
    
    # best model
    best_model = algorithm().set_params(**best_params)
    
    model = best_model.fit(X_train,y_train)
    
    # test performance
    test_prediction = model.predict(X_test)
    
    
    # test performance
    output_test = get_performances(y_pred=test_prediction,y_true=y_test,return_dict=True)
    output_test = pd.DataFrame(output_test).T.reset_index().rename(columns={"index":"Metric",0:"Score"})
    output_test['Set'] = 'Test'
    
    # train performance
    train_prediction = model.predict(X_train)
    
    output_train = get_performances(y_pred=train_prediction,y_true=y_train,return_dict=True)
    output_train = pd.DataFrame(output_train).T.reset_index().rename(columns={"index":"Metric",0:"Score"})
    output_train['Set'] = 'Train'
    
    # concatenate the train and test performance
    output = pd.concat([output_train,output_test])
    
    # add the algorithm, imputer, and balancer names
    output['Algorithm'] = algorithm_name
    output['ImputerCat'] = cat_imputer_name
    output['ImputerNum'] = num_imputer_name
    output['Imbalance'] = balancer_name    
    
    
    # reorganize the columns
    output = output[['Algorithm','Imbalance','ImputerCat','ImputerNum','Set','Metric','Score']]


    # save the best model
    model_path = os.path.join("../models")
    if not os.path.exists(model_path):
        
        os.makedirs(model_path)
       
       
    # save the best model 
    dump(model,os.path.join(model_path,algorithm_name+"_best_model.pkl"))


    # save the results    
    if not os.path.exists(dest_folder):
        
        os.makedirs(dest_folder)

      
    # save performance metrics
    output.to_csv(os.path.join(dest_folder,algorithm_name+"_best_model_perfomance.csv"),index=False)    
    
    # save best hyperparameters
    with open(os.path.join(dest_folder,algorithm_name+"_best_model_params.json"), 'w') as file:
        
        json.dump(best_params,file)
        
        
    print(f"{algorithm_name} hyperparameter tuning was done!!")
    
    return best_model,best_params,output





def train_best_models(df=None,data_path=None):
    
    
    """
    This function iterates over all combinations of algorithms, imputers, and balancers and finds the best model for each combination.
    
    The function also combines all results in a csv file.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The data. The default value is None.
    data_path: str
        The path to the data. The default value is None.
        If both df and data_path are None, the function will raise an error.
        
    Returns:
    --------
    None
        
    """
    
    df = pd.read_csv(data_path) if df is None else df   
    
    #  get combinations of model,balance,imputer
    combinations = get_combinations()

    
    best_results_for_all_models = None
  
    for combination in combinations:

        algorithm, imputer,balanc = combination

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
        
    
        

    
    