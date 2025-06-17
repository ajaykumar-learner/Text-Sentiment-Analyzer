import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER data if not already downloaded
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive 😊'
    elif compound_score <= -0.05:
        sentiment = 'Negative 😞'
    else:
        sentiment = 'Neutral 😐'
    return sentiment, scores

# Streamlit UI
st.title("📊 Text Sentiment Analyzer (NLP Project)")
st.write("Enter your sentence below to analyze the sentiment.")

user_input = st.text_area("Your Text", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        sentiment, scores = analyze_sentiment(user_input)
        st.markdown(f"### Sentiment: {sentiment}")
        st.subheader("Detailed Scores")
        st.json(scores)

