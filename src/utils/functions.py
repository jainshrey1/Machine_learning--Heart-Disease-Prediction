from fnmatch import fnmatch
import os
import pandas as pd



def concat_results(root='../results/',dest_path='../results/general/best_results_all_models.csv'):
    
    
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


if __name__ == "__main__":
    
    concat_results()
    
    print("All results are concatenated and saved to ../results/general/best_results_all_models.csv")