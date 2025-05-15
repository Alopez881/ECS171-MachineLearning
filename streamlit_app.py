import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Alopez881/ECS171-MachineLearning/refs/heads/master/reviews.csv')
  df

  st.write('**X**')
  X = df.drop('score', axis=1)
  X

  st.write('**y**')
  y = df.score
  y
