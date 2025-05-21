import streamlit as st
import pandas as pd
import joblib

# Load the trained sentiment model
model = joblib.load('logreg_sentiment.pkl')  # Make sure this file is in the same directory

st.title('App Review Sentiment Detector')
st.info('Write a review about an app and this model will predict whether it is **positive** or **negative**.')

# Text input for user review
user_input = st.text_area("Write your app review here:")

# Predict sentiment when the user inputs text
if user_input:
    prediction = model.predict([user_input])[0]
    sentiment = 'Positive 😊' if prediction == 1 else 'Negative 😞'
    st.subheader(f"Predicted Sentiment: **{sentiment}**")
