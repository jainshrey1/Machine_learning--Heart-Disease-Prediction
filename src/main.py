from utils.search_model import train_model
import pandas as pd
from utils.models import find_best_model
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    
    # df = pd.read_csv('../data/initial_data/frmgham2_project_data.csv')
    
    # X,y = 
    
    # train_model("../data/initial_data/frmgham2_project_data.csv",'CVD','../src/results/general/first_attempt_result_4_balancers.csv')
    
    
    # make_plots("../src/results/first_attempt_result.csv")
    
    path_to_save = "./results/log_reg/"
    

    
    find_best_model(algorithm=LogisticRegression,
                        data_path="../../data/initial_data/frmgham2_project_data.csv",
                        balancer=SMOTE,test_size=.2,KNNImputer = KNNImputer(),SimpleImputer = SimpleImputer(strategy='most_frequent'))
    
    
    
    