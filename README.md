# “Heart Disease Prediction Using Machine Learning” 
## Abstract
In the “Heart Disease Prediction Using Machine Learning” research project, our team investigates methods to accurately
classify individuals as either having heart disease or being healthy. The study leverages various classification algorithms, including
decision trees, support vector machines (SVM), logistic regression, Na ̈ıve Bayes, and k-nearest neighbors (K-NN). Ensemble
techniques such as random forests, gradient boosting, bagging, and XGBoost enhance model performance. The team utilizes several
sampling techniques to address the class imbalance: Synthetic Minority Over-sampling Technique (SMOTE), MWMOTE, ADASYN, for
oversampling, and AllKNN for undersampling. SMOTE generates synthetic instances of the minority class. Handling missing values is
crucial for robust model training. We employed KNN and Simple imputer techniques to fill in missing data points. These imputation
methods estimate missing values based on the similarity of neighboring data points and descriptive statistics. The comprehensive
approach aims to improve heart disease prediction accuracy, contributing valuable insights for healthcare professionals and
policymakers. Ultimately, the research findings can potentially enhance patient care and outcomes.

## Project structure

- **[data/](data/)**
   - **[initial_data/](data/initial_data/)**:  Full data set for modeling
   - **[model_data/](data/model_data/)**    :  Period 3 data set
- **[src/](src/)**
   - **[models/](src/models/)**   :  The best models for each algorithm in .pkl format
   - **[results/](src/results/)** :  Performance results for each algorithm and total
   - **[utils/](src/utisl/)**     :  Utility functions package
      - **[__init__.py](src/utils/__init__.py)**:  Python package initialization
      - **[data_preparation.py](src/utils/data_preparation.py)**:  Utility function for data preparation
      - **[functions.py](src/utils/functions.py)**:  Utility function for best results combination
      - **[get_parameters.py](src/utils/__init__.py)**:  Utility functions for parameters' combination collection 
      - **[metrics.py](src/utils/metrics.py)**:  Utility function to evaluate model performance
      - **[search_model.py](src/utils/search_model.py)**:  Utility function to train default models with several combinations
      - **[visual.py](src/utils/visual.py)**:  Utility functions to plot models' performance results
      - **[models.py](src/utils/models.py)**:  Utility functions to train 'the best' model for each algorithm
   - **[main.py](src/main.py)**:  The main file in a project
   - **[eda-1.ipynb](src/eda-1.ipynb)**:  Jupyter Notebook for EDA
   - **[end_to_end.ipynb](src/end_to_end.ipynb)**:  Jupyter Notebook for end-to-end workflow presentation
   - **[visualizations.ipynb](src/visualizations.ipynb)**:   Jupyter Notebook for visualizations

- **[requirements.txt](/requirements.txt)**: The inclusion of a requirements.txt file makes it easier to recreate the project's environment and install the necessary dependencies.
