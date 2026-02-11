# Flipkart_Sentiment_Analysis

This is a real-time sentiment analysis web app that classifies Flipkart product reviews as Positive or Negative using NLP and machine learning. The goal is simple — help understand customer feedback quickly and clearly.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

About the Project

Online reviews matter a lot. They influence buying decisions and help businesses improve their products. This project analyzes Flipkart product reviews and predicts whether the sentiment behind a review is positive or negative.

The app is built using Streamlit and runs a trained machine learning model in the background to make instant predictions.

What the App Can Do

Predict sentiment in real time

Simple and clean Streamlit interface

Uses a pre-trained ML model

TF-IDF for text feature extraction

Binary classification (Positive / Negative)

Clear results with emojis and color indicators

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Tech Stack

Python 3.x

Streamlit

scikit-learn

pandas

numpy

pickle

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Project Structure
Flipkart-Sentiment-Analysis/

├── sentiment_app.py

├── sentiment_model.pkl

├── tfidf_vectorizer.pkl

├── sentimental_analysis.ipynb

├── data.csv

├── requirements.txt

└── README.md

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

app.py – Streamlit web application

sentiment_model.pkl – Trained model

tfidf_vectorizer.pkl – Saved TF-IDF vectorizer

sentimental_analysis.ipynb – Model training notebook

data.csv – Dataset used

requirements.txt – Required Python libraries

#📊 Model Information
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Algorithm: Logistic Regression or Naive Bayes

Text processing: TF-IDF (Term Frequency – Inverse Document Frequency)

Output: Positive or Negative

Dataset: Flipkart product reviews

# How It Works
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The user enters a review.

The text is cleaned and prepared.

TF-IDF converts text into numerical features.

The trained model predicts the sentiment.

The result is shown instantly on the screen.

# 🎯Example
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Positive Review

Input:
“This product is amazing! Great quality and fast delivery.”

Output:
Positive 😊

Negative Review

Input:
“Very bad product. Quality is poor and delivery was delayed.”

Output:
Negative 😞

# Model Training
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The training process included:

Collecting review data

Cleaning and preprocessing text

Converting text to features using TF-IDF

Training a classification model

Saving the model using pickle

You can see the full training workflow inside sentimental_analysis.ipynb.

# Deployment
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This app is deployed using Streamlit Cloud so it can be accessed easily from anywhere.

If you want to deploy your own version:

Fork the repository

Sign up on Streamlit Cloud

Connect your GitHub repo

Deploy

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Author

Saiteja
GitHub:
