import pandas as pd
import matplotlib.pyplot as plt



def make_plots(df_path):
    
    
    df = pd.read_csv(df_path)
    
    
    algorithms = df['Algorithm'].unique().tolist()
    
    
    for algorithm in algorithms:
        
        alg_df = df[df['Algorithm']==algorithm]
        
    
    print(algorithms)