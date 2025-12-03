# Movie-Review-Sentiment-Analysis
A machine learning project that predicts movie review sentiment (positive/negative) using Logistic Regression, Random Forest and Linear SVC. Includes complete data preprocessing, model training, evaluation metrics and a Tkinter GUI for real-time review prediction.
Project Features

Text preprocessing (cleaning, stopwords removal, TF-IDF)

Multiple ML models:

Logistic Regression

Random Forest

Linear SVC


Evaluation metrics:

Accuracy, Precision, Recall, F1-Score, Confusion Matrix


Exports evaluation results to evaluation_results.xlsx

Simple GUI to input a review and get prediction from all models


Accuracy Summary

Logistic Regression: ~78%

Random Forest: ~80%

Linear SVC: ~80%


How to Run

1. Install required libraries:

pip install pandas numpy scikit-learn nltk tkinter openpyxl

2. Run the application:

python main.py

3. A GUI window opens → enter any movie review → click Predict Sentiment


Technologies Used

Python
Scikit-Learn
NLTK
Pandas & NumPy
Tkinter
TF-IDF Vectorizer


What You Learn

NLP basics
Text preprocessing
ML model training & evaluation
Integrating ML with GUI applications
