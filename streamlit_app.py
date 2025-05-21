import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('new_sentiment_model.pkl')  # Make sure this file is in the same folder

# Streamlit UI
st.title('App Review Sentiment Classifier')
st.info('Type a review of a mobile app and the model will predict if it is **positive** or **negative**.')

# Text input
user_input = st.text_area("Enter your app review:")

# Prediction
if user_input:
    prediction = model.predict([user_input])[0]
    label = 'Positive 😊' if prediction == 1 else 'Negative 😞'
    st.success(f'The review is predicted to be: **{label}**')
