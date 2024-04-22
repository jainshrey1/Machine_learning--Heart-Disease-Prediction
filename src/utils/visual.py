import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *


#  performance for each algorithm

def plot_for_each_algorithm(df):


    algoritms = df.Algorithm.unique()

    algoritms_title = {"LogisticRegression":"Performance of Logistic Regression for each impution and data balancing approach",
                  "DecisionTree":"Performance of Decision Tree for each impution and data balancing approach",
                  "NaiveBayes":"Performance of Naive Bayes for each impution and data balancing approach",
                  "KNN":"Performance of KNN for each impution and data balancing approach",
                  "SVM":"Performance of SVM for each impution and data balancing approach"}
    
    for alg in algoritms:
    
        alg_df = df[df['Algorithm'] == alg]
        
        bar = ggplot(alg_df, aes(x='Metric', y='Score', fill='Metric',label='Score')) + \
               facet_grid(rows='Imbalance',cols='Imputer') + \
               geom_col(position='dodge') + \
               theme_minimal() + \
               theme(figure_size=(15,15),axis_text_x=element_text(angle=45, hjust=1))  + \
               geom_text(position=position_dodge(width=.3),va='bottom') +\
               ylim(0,110) + \
               labs(title=algoritms_title[alg])
    
    
        bar.show()
        # break
        
        
        # for each balancing techqnique

def plot_for_each_balancer(df):


    balancers = df.Imbalance.unique()

    
    for bal in balancers:
    
        bal_df = df[df['Imbalance'] == bal]
        # return bal_df

        title = f"Performances of algoriths with {bal} balancing approach"

        if bal == 'OriginalData':
            title = f"Performances of algoriths with original data"
        
        bar = ggplot(bal_df, aes(x='Metric', y='Score', fill='Metric',label='Score')) + \
               facet_grid(rows='Algorithm',cols='Imputer') + \
               geom_col(position='dodge') + \
               theme_minimal() + \
               theme(figure_size=(15,15),axis_text_x=element_text(angle=45, hjust=1))  + \
               geom_text(position=position_dodge(width=.3),va='bottom') +\
               ylim(0,110) + \
               labs(title=title)
    
    
        bar.show()
        # break

def plot_for_each_performance_metric(df):

    metrics = df.MainMetric.unique()
    
    for met in metrics:
    
        met_df = df[df['MainMetric'] == met]
    
        title = f"Train and test comparison based on {met}"
    
        bar = ggplot(met_df, aes(x='Algorithm', y='Score', fill='Metric',label='Score')) + \
                   facet_grid(rows='Imbalance',cols='Imputer') + \
                   geom_col(position='dodge') + \
                   theme_minimal() + \
                   theme(figure_size=(15,15),axis_text_x=element_text(angle=45, hjust=1))  + \
                   geom_text(position=position_dodge(width=1),va='bottom',ha='center') +\
                   ylim(0,110) + \
                   labs(title=title)
        
        bar.show()
