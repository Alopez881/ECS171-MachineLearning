import streamlit as st
import pandas as pd
import joblib
from collections import Counter

# --- Page Setup ---
st.set_page_config(page_title="App Review Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 App Review Sentiment Analyzer")
st.markdown("Type a review and this app will predict whether it's **positive** or **negative**.")

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("logreg_sentiment.pkl")

model = load_model()

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_reviews.csv")

df = load_data()

# --- Chat Box Form ---
with st.form("chat_form"):
    review = st.text_area("📨 Your app review:", height=150, placeholder="Write your review here...")
    submitted = st.form_submit_button("Analyze Sentiment")

# --- Prediction Output ---
if submitted:
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        input_data = {
            'cleaned_text': [review],
            'has_upvotes': [0],
            'review_length': [len(review.split())],
            'exclamation_count': [review.count('!')],
            'question_count': [review.count('?')],
            'ngram_great app': [0],
            'ngram_good app': [0],
            'ngram_easy use': [0],
            'ngram_love app': [0],
            'ngram_pro version': [0],
            'ngram_google calendar': [0],
            'ngram_free version': [0],
            'ngram_use app': [0],
            'ngram_like app': [0],
            'ngram_doesnt work': [0],
            'ngram_really like app': [0],
            'ngram_app easy use': [0],
            'ngram_buy pro version': [0],
            'ngram_using app years': [0],
            'ngram_paid pro version': [0],
            'ngram_really good app': [0],
            'ngram_simple easy use': [0],
            'ngram_used app years': [0],
            'ngram_sync google calendar': [0],
            'ngram_todo list app': [0]
        }

        input_df = pd.DataFrame(input_data)
        pred = model.predict(input_df)[0]
        label = "Positive 😊" if pred == 1 else "Negative 😞"
        st.success(f"**Sentiment:** {label}")

# --- Sentiment Distribution ---
with st.expander("📊 Sentiment Distribution"):
    sentiment_counts = df["sentiment_binary"].value_counts().rename({0: "Negative", 1: "Positive"})
    st.bar_chart(sentiment_counts)

# --- Top Words Table ---
def get_top_words(series, n=10):
    words = " ".join(series).split()
    top = Counter(words).most_common(n)
    return pd.DataFrame(top, columns=["Word", "Count"])

with st.expander("📄 Most Common Words by Sentiment"):
    st.markdown("**Top words in Positive Reviews**")
    st.dataframe(get_top_words(df[df.sentiment_binary == 1]["cleaned_text"]))
    st.markdown("**Top words in Negative Reviews**")
    st.dataframe(get_top_words(df[df.sentiment_binary == 0]["cleaned_text"]))

# --- Top Predictive Words (Static Placeholder) ---
with st.expander("🧠 Top Words That Predict Sentiment"):
    st.write("These are placeholder words from logistic regression results.")
    st.dataframe(pd.DataFrame({
        "Positive Words": ["love", "great", "easy", "helpful", "amazing"],
        "Negative Words": ["useless", "uninstalled", "doesn't", "buggy", "complicated"]
    }))

# --- Sentiment Over Time ---
with st.expander("📅 Sentiment Over Time"):
    df['at'] = pd.to_datetime(df['at'], errors='coerce')
    sentiment_by_day = df.dropna(subset=['at']).groupby(df['at'].dt.date)['sentiment_binary'].mean()
    st.line_chart(sentiment_by_day)

# --- Review Browser ---
with st.expander("🔍 Browse Reviews by Sentiment"):
    choice = st.radio("Show reviews that are...", ["All", "Positive", "Negative"])
    if choice == "All":
        st.dataframe(df[(df.sentiment_binary == 0) | (df.sentiment_binary == 1)][["content", "score"]].head(250))
    elif choice == "Negative":
        st.dataframe(df[df.sentiment_binary == 0][["content", "score"]].head(20))
    elif choice == "Positive":
        st.dataframe(df[(df.sentiment_binary == 1) | (df.sentiment_binary == 2)][["content", "score"]].head(250))
        
