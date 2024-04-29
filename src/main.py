"""
This is the main file to run the pipeline.

The pipeline consists of the following steps:
    - Train default models
    - Train the best models
    
The visualizations are not part of the pipeline, but the functions are in the src/utils/visual.py file, and plots are in visualizations.ipynb notebook.
"""

from utils.search_model import train_models
import pandas as pd
from utils.models import train_best_models


from utils.get_parameters import get_combinations



if __name__ == "__main__":
    
    # read the data
    data_path ='../data/initial_data/frmgham2_project_data_full.csv'
    df = pd.read_csv(data_path)
    
    # train defualt models
    performarmace_df = train_models(df=df,target_var='CVD',path_to_save='../src/results/general/full_data_performances_9_models_5_balancers.csv')
    

    # train the best models
    # all intermidiate results are saved and functions will use them to train the best models
    train_best_models(df=df)

    
    
    
    
    