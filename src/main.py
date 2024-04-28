from utils.search_model import train_models
import pandas as pd
from utils.models import find_best_model,train_best_models
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from imblearn.under_sampling import ClusterCentroids


from utils.get_parameters import get_combinations



if __name__ == "__main__":
    
    
    # train defualt models
    # performarmace_df = train_models("../data/initial_data/frmgham2_project_data_full.csv",'CVD','../src/results/general/full_data_performances_9_models_5_balancers.csv')
    
    # data_path ='../data/initial_data/frmgham2_project_data_full.csv'
    # df = pd.read_csv(data_path)

    train_best_models()

    # get combinations of model,balance,imputer
    # combinations = get_combinations()
    
    # # print(combinations)
    # # quit()
    
    # df = pd.read_csv('../data/initial_data/frmgham2_project_data_full.csv')
  
    # for combination in combinations:

    #     algorithm, imputer,balanc = combination
    #     # print(combination)
    #     # quit()
    #     print(algorithm.__name__,balanc.__name__,imputer)

    #     best_model,best_params,output = find_best_model(algorithm=algorithm,
    #                                                 balancer=balanc,
    #                                                 imputer=imputer,
    #                                                 df=df,
    #                                                 ovewrite=True)

        # break
    
    
    
    