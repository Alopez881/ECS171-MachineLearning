import pandas as pd
import streamlit as st
import joblib

# Load model
model = joblib.load('logreg_sentiment.pkl')  # Trained model (pipeline)

st.title('App Review Sentiment Detector')
st.info('Write a review about an app and this model will predict whether it is **positive** or **negative**.')

# Get input
user_input = st.text_area("Write your app review here:")

# Predict
if user_input:
    input_df = pd.DataFrame({'content': [user_input]})  # Replace 'content' if your model expects another column name
    prediction = model.predict(input_df)[0]
    sentiment = 'Positive 😊' if prediction == 1 else 'Negative 😞'
    st.subheader(f"Predicted Sentiment: **{sentiment}**")
