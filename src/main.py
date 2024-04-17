from utils.modeling import train_model
import pandas as pd


if __name__ == "__main__":
    
    df = pd.read_csv('../data/model_data/period_3_rel_features_df_by_tree.csv')
    
    # X,y = 
    
    train_model("../data/model_data/period_3_rel_features_df_by_tree.csv",'CVD','../src/results/first_attempt_result.csv')