import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info('This is app buildis a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read.csv('https://raw.githubusercontent.com/Nissi-d/sentiment-analysis/refs/heads/main/data/raw/reviews.csv')
  df
