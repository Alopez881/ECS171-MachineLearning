import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page title
st.title('Restaurant Review Sentiment Classifier')

st.info('Type in a review to find out if it’s **positive** or **negative** based on real user data.')

# Load and prepare data
df = pd.read_csv('https://raw.githubusercontent.com/Alopez881/ECS171-MachineLearning/refs/heads/master/reviews.csv')  # make sure 'reviews.csv' is in the same directory if running locally

# Show raw data
with st.expander('Raw Data Preview'):
    st.write(df.head())

# Assume columns are 'content' (text) and 'score' (label: 1 for positive, 0 for negative)
X = df['content']
y = df['score']

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Review input
st.subheader('Try it out:')
user_input = st.text_area("Write a fake restaurant review here:")

if user_input:
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    label = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.success(f"This review is predicted to be: **{label}**")
