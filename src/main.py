from utils.search_model import train_models
import pandas as pd
from utils.models import find_best_model
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from imblearn.under_sampling import ClusterCentroids


if __name__ == "__main__":
    
    # df = pd.read_csv('../data/initial_data/frmgham2_project_data.csv')
    
    # X,y = 
    
    train_models("../data/initial_data/frmgham2_project_data_full.csv",'CVD','../src/results/general/full_data_performances_9_models_5_balancers.csv')
    
    quit()
    
    # make_plots("../src/results/first_attempt_result.csv")
    
    path_to_save = "./results/log_reg/"
    

    
    # find the best model for each algorithm based on test results
    algoritms = [DecisionTreeClassifier,KNeighborsClassifier,LogisticRegression,SVC]
    balancers = [ClusterCentroids,SMOTE,SMOTE,SMOTE]
    imputation = [['SimpleImputer_mean', SimpleImputer(),'SimpleImputer_mode', SimpleImputer(strategy='most_frequent')],
                ['KNNImptuer',KNNImputer(),'SimpleImputer_mode',SimpleImputer(strategy='most_frequent')],
                ['KNNImptuer',KNNImputer(),'SimpleImputer_mode',SimpleImputer(strategy='most_frequent')],
                ['SimpleImputer_mean', SimpleImputer(),'SimpleImputer_mode', SimpleImputer(strategy='most_frequent')]]
    data_path="../../data/initial_data/frmgham2_project_data.csv"

    for algorithm, balanc, imputer in zip(algoritms,balancers,imputation):

        print(algorithm.__name__,balanc.__name__,imputer)

        best_model,best_params,output = find_best_model(algorithm=algorithm,
                                                    data_path=data_path,balancer=balanc,imputer=imputer)


    
    
    
    