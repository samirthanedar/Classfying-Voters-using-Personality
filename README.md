# Classifying Voters using Personality

## Overview

Political campaigns have to market their candidates like any company advertising a product. And like companies, they know their candidates will appeal more to a certain group of people. For instance, for a primary election you might only want to target voters who will vote in your party’s primary election. In a general election, you'll probably want to know which states are toss-ups so you spend the majority of your time there. For instance, a democratic candidate for president might not want to spend a lot of time in California because they should win easily there. 

In order to understand how people will vote, political campaigns mostly use polling. This is an attempt to predict voting outcomes without polling. 


## Objective

My goal is to create a classification model that will predict who someone will vote for based on some demographic information and their answers to informal personal questions. I used a [dataset from Kaggle](https://www.kaggle.com/c/can-we-predict-voting-outcomes/data) to build this model. 


## Features

The dataset has two types of features:
1. Demographic info like Age, Gender, Education level, Marital status, and Income
1. Yes or No answers to personal questions (e.g. Are you an idealist or a pragmatist?)

## Findings:

First, I found that three classification models performed the best: Naive Bayes Bernoulli, Logistic Regression, and Random Forest. Random Forest had the highest AUC score and F1 score (67%) but was overfitting. The next best model was an ensemble model combining all three of the models I mentioned above. Overall, I think the Ensemble model is probably the best since it is not as overfit and only has a slightly lower F1 score(66%). 


## Navigating the Project Files:

You should follow this workflow when going through my notebooks to repeat the results: 
1. Start at 01_EDA.ipynb to see the initial EDA I did
1. Next go to 02_model_selection.ipynb to see how I went through several classification models until I found the three that were consistently performing the best
1. Next go to 03_feature_engineering.ipynb to see all the ways I tried to engineer features.
1. Finally, go to 04_model_tuning.ipynb to see my final hyperparameter tuning 

I've also included the following files to help you reproduce the results
* Pickle file of the cleaned dataset: ready.pkl
* Raw dataset: train2016.csv
* Tuned models in the models folder
* Flask app files in web_app_files

## Web App

Finally, please check out the [web app](https://tranquil-falls-00014.herokuapp.com/predict) I created. I've put the top 12 features as questions. You can answer them and see if the model predicts your political party of choice corrrectly!


