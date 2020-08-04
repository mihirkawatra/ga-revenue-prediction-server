# Google Analytics Customer Revenue Prediction

Kaggle Challenge Link: https://www.kaggle.com/c/ga-customer-revenue-prediction/submissions


| **Title** | **Description** | **Link** |
|--|--|--|
| Data Wrangling, EDA & Visualization | Data Loading, Flattening of JSON Fields, EDA, and Plotting to get an overview of the data we have. | [https://www.kaggle.com/mihirkawatra/gacrp-eda-and-visualization](https://www.kaggle.com/mihirkawatra/gacrp-eda-and-visualization) |
| Data Preprocessing, Feature Engineering, PCA and Clustering | Cleaning and Preprocessing of unstructured and messy data to prepare it for modeling, followed by PCA and K-Means Clustering | [https://colab.research.google.com/drive/1AfLk65shVrHxaQJrXXS1rYD2JTAH3YJa?usp=sharing](https://colab.research.google.com/drive/1AfLk65shVrHxaQJrXXS1rYD2JTAH3YJa?usp=sharing) |
| Modeling | Model Testing using the preprocessed data from the previous file, Hyperparameter Tuning to get the optimized parameters. | [https://colab.research.google.com/drive/1DQPFJqlpi6wpSVchTjnSvwphpg69fi0G?usp=sharing](https://colab.research.google.com/drive/1DQPFJqlpi6wpSVchTjnSvwphpg69fi0G?usp=sharing) |

## Setting Up
 - `git clone https://github.com/mihirkawatra/ga-revenue-prediction-server.git`
 - `conda create -n flask-env`
 - `source activate flask-env`
 - `pip install -r requirements.txt`

## Steps to run
 - `cd ga-revenue-prediction-server`
 - `python app.py`
 
      *OR*
 - `flask run`
