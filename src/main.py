from utils.modeling import train_model
from utils.visual import make_plots
import pandas as pd


if __name__ == "__main__":
    
    # df = pd.read_csv('../data/initial_data/frmgham2_project_data.csv')
    
    # X,y = 
    
    train_model("../data/initial_data/frmgham2_project_data.csv",'CVD','../src/results/first_attempt_result_4_balancers.csv')
    
    
    # make_plots("../src/results/first_attempt_result.csv")