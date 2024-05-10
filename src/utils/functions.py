"""
This is a file for any addtional functions that are not part of the main pipeline.
"""

from fnmatch import fnmatch
import os
import pandas as pd
import json



def concat_results(root='../results/',dest_path='../results/general/best_results_all_models.csv'):
    
    
    """
    Takes the root where the results are stored and concatentas all the results into one file.
    Since the results are stored in different folders, this function goes through all the folders and reads the csv files using fnmatch.
    
    
    Parameters:
    -----------
    root: str
        The root where the results are stored. The default value is '../results/'
        
    dest_path: str
        The path where the concatenated results will be saved. The default value is '../results/general/best_results_all_models.csv'
        
    Returns:
    --------
    pd.DataFrame
        The concatenated results.
    
    """
    
    full_df = None
    # root = "./results/"
    for path, _, files in os.walk(root):

        if path.split("/")[-1] == 'general':
            continue
        for name in files:
            if fnmatch(name, "*.csv"):
                file_path = os.path.join(path,name)

                if full_df is None:
                    full_df = pd.read_csv(file_path)
                    continue

                full_df = pd.concat([full_df,pd.read_csv(file_path)])
    
    
    full_df = full_df[full_df['Metric'].isin(['Accuracy', 'F-1', 'Kappa', 'Recall', 'Precision','AUC'])]
    full_df.to_csv(dest_path,index=False)
    

    return full_df




def get_best_params(path):
    
    """
    Reads the best parameters from the JSON file. 
    Converts in into DataFrame and returns it.
    
    Parameters:
    -----------
    path: str
        The path to the JSON file.
        
    Returns:
    --------
    pd.DataFrame
        The best parameters.
    """
    
    
    with open(path,'r') as f:
        
        best_params = json.load(f)
        
        best_params = {k:[v] for k,v in best_params.items()}    
        
        return pd.DataFrame(best_params).T.reset_index().rename(columns={'index':'Parameter',0:'Value'})
    
if __name__ == "__main__":
    
    concat_results()
    
    print("All results are concatenated and saved to ../results/general/best_results_all_models.csv")