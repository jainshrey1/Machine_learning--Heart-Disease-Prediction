"""
This files contains the functions to calculate model evaluation metrics.
"""

import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,accuracy_score,cohen_kappa_score




# lambda functions to calculate false positive rate and false negative rate
false_positive_rate = lambda fp,tn: round((fp)/(fp+tn),2) # number of false positives devided by total number of negatives
false_negative_rate = lambda fn,tp: round((fn)/(fn+tp),2) # number of false negatives devided by total number of positives
 

def get_performances(y_true,y_pred,return_dict = False):
    
    
    """
    This function calculates the model evaluation metrics.
    The metrics are:
        - Accuracy
        - F1
        - True Negative
        - False Positive
        - False Negative
        - True Positive
        - Kappa
        - Recall
        - Precision
        - False Postive Rate
        - False Negative Rate
        
        
    Parameters:
    -----------
    y_true: np.array
        The true labels.
    y_pred: np.array
        The predicted labels.    
    return_dict: bool
        If True, the function returns a dictionary with the metrics as keys. The default value is False.
        
    Returns:
    --------
    list or dict
        A list with the metrics values or a dictionary with the metrics as keys.
    
    
    """
    acc = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    conf_matrix = confusion_matrix(y_true,y_pred)
    tn, fp, fn, tp = [int(c) for c in conf_matrix.ravel()]
    kappa_score = cohen_kappa_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    fpr = false_positive_rate(fp,tn)
    fnr = false_negative_rate(fn,tp)    


    output = list(map(lambda x: round(x*100,2),[acc,f1,tn/100,fp/100,fn/100,tp/100,kappa_score,recall,precision,fpr/100,fnr/100]))
    
    if return_dict:
        
        metrics = ['Accuracy','F-1','True Negative','False Positive','False Negative','True Positive','Kappa','Recall','Precision','False Postive Rate','False Negative Rate']
        
        output = {metrics[i]:[output[i]] for i in range(len(output))}
        
    
    return output