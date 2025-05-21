import streamlit as st
import joblib

# --- Styling ---
st.set_page_config(page_title="App Review Sentiment", page_icon="📱", layout="centered")

# --- Custom CSS for style ---
st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
    }
    .center-text {
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 18px;
        padding: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 0.5em 2em;
        border-radius: 8px;
        margin-top: 1em;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='center-text'>📱 App Review Sentiment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='big-font center-text'>Paste a review below and find out if it's positive or negative.</p>", unsafe_allow_html=True)

# --- Input Area ---
user_input = st.text_area("Write your app review here 👇")

# --- Button and Prediction Section (Logic Placeholder) ---
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before clicking Analyze.")
    else:
        # Placeholder result (you can connect your model later)
        st.markdown("---")
        st.subheader("🔍 Predicted Sentiment:")
        st.success("Positive 😊")  # ← swap with model result later
        st.caption("This is a placeholder result. Model output will appear here once connected.")

# --- Future Section for Model Details, Uploads, or Charts ---
with st.expander("📊 Want to see the data or upload your own?"):
    st.write("This space is reserved for displaying review datasets or letting users upload their own CSV files.")
    st.button("Upload Dataset (Coming Soon)")
