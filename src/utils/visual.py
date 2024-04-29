""":
To make visualization process easier and simple, we have coded functions for each type of visualization.
Here plotnine and matplotlib packages are used to create the plots. plotnine is a Python package allowing to create 'ggplot2' like plots.


"""

from plotnine import *
import matplotlib.pyplot as plt
from utils.get_parameters import max_score_for_each

#  performance for each algorithm

def plot_for_each_algorithm(df):


    """
    This function plots the performance of each algorithm for each impution and data balancing approach.
    The function outputs a plot for each algorithm where
        - x-axis: the metric
        - y-axis: the score
        - color: the metric
        - facet_grid: the imputer and the data balancing approach
            - row: the balancing technique
            - column: the imputer
    
    
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the results.
        
    Returns:
    --------
    None
    """

    algoritms = df.Algorithm.unique()


    
    for alg in algoritms:
    
        alg_df = df[df['Algorithm'] == alg]
        
        title = f"Performance of {alg} for each impution and data balancing approach" 
        
        bar = ggplot(alg_df, aes(x='Metric', y='Score', fill='Metric',label='Score')) + \
               facet_grid(rows='Imbalance',cols='Imputer') + \
               geom_col(position='dodge') + \
               theme_minimal() + \
               theme(figure_size=(15,15),axis_text_x=element_text(angle=45, hjust=1))  + \
               geom_text(position=position_dodge(width=.3),va='bottom') +\
               ylim(0,110) + \
               labs(title=title)
    
    
        bar.show()
        # break
        
        
        # for each balancing techqnique

def plot_for_each_balancer(df):


    """
    This function plots the performance of each balancing technique for each impution and algorithm.
    This function outputs a plot for each balancing technique where
        - x-axis: the metric
        - y-axis: the score
        - color: the metric
        - facet_grid: the imputer and the algorithm
            - row: the algorithm
            - column: the imputer
            
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the results.
        
    Returns:
    --------
    None

    """

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


    """
    The function plots the performance of each metric for each imputation, balancing and algorithm combination.
    
    This function outputs a plot for each metric where
        - x-axis: the algorithm
        - y-axis: the score
        - color: the metric/set
        - facet_grid: the imputer and the data balancing approach
            - row: the balancing technique
            - column: the imputer
    
    
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the results.
        
    Returns:
    --------
    None
    """
    
    df['Score'] = round(df['Score'],1)
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



def plot_for_each_best_algorithm(df,set_='Test'):
    
    """
    This function plots the performance of each algorithms' best model.
    
    The rows * columns = number of algorithms.
    Each subplot show the performance of an algorithm.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the results.
        
    set_: str
        The set to plot. The default value is 'Test'
        
        
    Returns:
    --------
    None
    """

    
    part = "Total"
    data = df.copy()
    
    if len(set_) != 0:

        try:
            data = df[df['Set'] == set_].copy()
            part = set_
        except:
            pass

    
    
    fig, ax = plt.subplots(3, 3)
    
    title = fig.suptitle(f"{part} scores for each algorithm", fontsize=16)
    
    # Adjust the position of the title
    title.set_position([.5, 0.95])
    
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.5)
    axes = ax.flatten()
    
    algorithms = data.Algorithm.unique()
    
    colors = ['orange','blue','red','green','purple','yellow','magenta','cyan','olive']
    
    for i in range(len(algorithms)):


        alg = algorithms[i]

        plot_df = data[data['Algorithm'] == alg]
        
        axes[i].bar(plot_df['Metric'], plot_df['Score'],color=colors[i])
        axes[i].set_title(f"({str(i+1)}) {alg}")
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis ticks
        axes[i].set_ylim(0,110)
        axes[i].set_ylabel("Score")
        for j, value in enumerate(plot_df.Score):  # Changed the variable name to j
            axes[i].annotate(str(value), xy=(j, value), ha='center', va='bottom')
    
    plt.show()




def plot_best_restults_each_metric(df,set_='Test'):
    
    
    """
    This function plots the best results for each metric.
    The rows * columns = number of metrics.
    Each subplot show the performances of algorithms for a metric. 
    The easiest way to compare the performance of algorithms for each metric.
    
    
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the results.
        
    set_: str
        The set to plot. The default value is 'Test'
        
    Returns:
    --------
    None
    
    """



    part = "Total"
    if len(set_) != 0:

        data = df[df['Set'] == set_].copy()
        part = set_
    # Create subplots
    fig, ax = plt.subplots(3, 2, figsize=(14, 11))  # Set the figure size
    
    
    title = fig.suptitle(f"{part} scores for each algorithm", fontsize=16)
    
    # Adjust the position of the title
    title.set_position([.5, 1.02])
                       
    fig.subplots_adjust(top=0.5)  # Adjust the spacing between subplots
    
    axes = ax.flatten()
    
    
    
    metrics = data.Metric.unique()
    colors = ['orange', 'blue', 'red', 'green', 'pink', 'yellow']
    
    for i, met in enumerate(metrics):
        
        plot_data = data[data['Metric'] == met].copy()
    
    
        axes[i].barh(plot_data['Algorithm'], plot_data['Score'], color=colors[i])
        
        for j, value in enumerate(data.Score):
            axes[i].annotate(str(value), xy=(value,j), ha='right', va='center')
    
        axes[i].set_title(f"({i + 1}) {met}")
        axes[i].invert_yaxis()
        axes[i].set_xlabel("Score")
    
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()
    
    
    
    
def plot_best_performed_counts(df,by=['Imputer','Imbalance','MainMetric'],set_='Test'):
    
    
    """
    The function plots the count for each algorithm that has the best performance.
    If an algorithm has the best performance all the time, the count will be n_metrics * n_imputers * n_balancers.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe with the results.
        
    by: list
        The columns to group by. The default value is ['Imputer','Imbalance','MainMetric']
        
    set_: str
        The set to plot. The default value is 'Test'
        
    Returns:
    --------
    pd.DataFrame
        The counts of the best performed algorithm.
    """
    
    overal_best_performed_test = max_score_for_each(df=df,by=by,set_=set_)
    
    counts_of_the_best_perfomred = overal_best_performed_test.Algorithm.value_counts().to_frame().reset_index()
    
    fig,ax = plt.subplots(1,1)

    fig.suptitle(f"The best scores counts for each algorithm on {set_} set ")
    
    ax.barh(counts_of_the_best_perfomred['Algorithm'], counts_of_the_best_perfomred['count'], color='orange')
    
    for j, value in enumerate(counts_of_the_best_perfomred['count']):
        ax.annotate(f"{str(value)} [{round(value/60*100,2)}%]", xy=(value,j), ha='left', va='center')
    
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    
    return counts_of_the_best_perfomred