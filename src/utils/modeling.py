from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import re

from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,accuracy_score,cohen_kappa_score


imputers_num = {"SimpleImputer_mean":SimpleImputer()}
imputers_cat = {"SimpleImputer_mean":SimpleImputer(strategy='most_frequent')}

imputers = [[["SimpleImputer_mean",SimpleImputer()],
            ["SimpleImputer_mode",SimpleImputer(strategy='most_frequent')]]]

balancers = {"SMOTE":SMOTE,
             "ClusterCentroids":ClusterCentroids,
             "OriginalData":None}

algorithms = {"LogisticRegression":LogisticRegression(),
          "DecisionTree":DecisionTreeClassifier(),
          "NaiveBayes":GaussianNB(),
          "KNN":KNeighborsClassifier(),
          "SVM":SVC()}


# metrics = {"accuracy":accuracy_score,
#            "f1score":f1_score,
#            "precision_score":precision_score,
#            "recall_score":recall_score,
#            "confusion_matrix":confusion_matrix
#            }

from copy import deepcopy

import pandas as pd


def train_model(path,target_var,path_to_save):
    
    performance_df = {"ImputerNum":[],
                      "ImputerCat":[],
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
                      "Test-F-1":[],
                      "Test-Accuracy":[],
                      "Test-Kappa":[],
                      "Test-TP":[],
                      "Test-FP":[],
                      "Test-TN":[],
                      "Test-FN":[],
                      "Test-Recall":[],
                      "Test-Precision":[],}
    
    
    df = pd.read_csv(path)
    
    
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
        
        # impute
        x_cat = cat_imputer.fit_transform(x_cat)
        x_num = num_imputer.fit_transform(x_num)
        
        # 
        ddf[cat_varaibles] = x_cat
        ddf[num_varaiables] = x_num
        
        X,y = ddf.iloc[:,:-1].values,ddf.iloc[:,-1].values
        
        # balance
        for b_name,balancer in balancers.items():
            
            
            
            if b_name != 'OriginalData':
                balancer = balancer()
                
                X,y = balancer.fit_resample(X,y)
            
            # train test and scale
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=123,test_size=.2)
            
            scaler = StandardScaler()
            
            scaler = scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            
            # train
            for m_name,model in algorithms.items():
                
                        
                performance_df['ImputerCat'].append(cat_imputer_name)
                performance_df['ImputerNum'].append(num_imputer_name)
        
                
                performance_df['Imbalance'].append(b_name)

                
                performance_df['Algorithm'].append(m_name)
                
                model = model.fit(X_train,y_train)
                
                y_pred = model.predict(X_train)
                
                # train
                acc = accuracy_score(y_train,y_pred)
                f1 = f1_score(y_train,y_pred)
                conf_matrix = confusion_matrix(y_train,y_pred)
                tn, fp, fn, tp = [int(c) for c in conf_matrix.ravel()]
                kappa_score = cohen_kappa_score(y_train,y_pred)
                recall = recall_score(y_train,y_pred)
                precision = precision_score(y_train,y_pred)
                
                performance_df['Train-Accuracy'].append(acc)
                performance_df['Train-F-1'].append(f1)
                performance_df['Train-Kappa'].append(kappa_score)
                performance_df['Train-Precision'].append(precision)
                performance_df['Train-Recall'].append(recall)
                performance_df['Train-TP'].append(tp) 
                performance_df['Train-TN'].append(tn)               
                performance_df['Train-FP'].append(fp)               
                performance_df['Train-FN'].append(fn)               
              
                # test
                
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test,y_pred)
                f1 = f1_score(y_test,y_pred)
                conf_matrix = confusion_matrix(y_test,y_pred)
                tn, fp, fn, tp = [int(c) for c in conf_matrix.ravel()]
                kappa_score = cohen_kappa_score(y_test,y_pred)
                recall = recall_score(y_test,y_pred)
                precision = precision_score(y_test,y_pred)
                
                performance_df['Test-Accuracy'].append(acc)
                performance_df['Test-F-1'].append(f1)
                performance_df['Test-Kappa'].append(kappa_score)
                performance_df['Test-Precision'].append(precision)
                performance_df['Test-Recall'].append(recall)
                performance_df['Test-TP'].append(tp) 
                performance_df['Test-TN'].append(tn)               
                performance_df['Test-FP'].append(fp)               
                performance_df['Test-FN'].append(fn)               
              
                
     
    performance_df = pd.DataFrame(performance_df).melt(id_vars=['Algorithm','ImputerNum','ImputerCat','Imbalance'],var_name='Metrics',value_name='Score')       
    performance_df.to_csv(path_to_save,index=False)
    # performance_df.
    return performance_df
            
            
            
            
        
        
    
    # if categorical, imputation techniques changes
    
    
    
    
    pass
    
    
    