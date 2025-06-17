import streamlit as st
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already present
nltk.download('vader_lexicon')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# --- Streamlit UI ---
st.set_page_config(page_title="Text Sentiment Analyzer", page_icon="📝", layout="centered")
st.title("📝 Text Sentiment Analyzer")

st.sidebar.header("Instructions")
st.sidebar.write(
    """
    - Enter any text (review, comment, tweet, etc.)
    - Click *Analyze Sentiment*
    - View sentiment classification & scores  
    - Choose between *TextBlob* or *VADER*
    """
)

# User input
text_input = st.text_area("Enter text below:", "")

# Sentiment method selection
method = st.sidebar.radio("Select sentiment analysis method:", ["TextBlob", "VADER"])

# Analyze button
if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("⚠ Please enter some text for analysis.")
    else:
        if method == "TextBlob":
            blob = TextBlob(text_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Determine sentiment
            if polarity > 0:
                sentiment = "Positive 😊"
                color = "green"
            elif polarity < 0:
                sentiment = "Negative 😞"
                color = "red"
            else:
                sentiment = "Neutral 😐"
                color = "gray"

            st.markdown(f"*Sentiment:* <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
            st.write(f"*Polarity Score:* {polarity:.2f}")
            st.write(f"*Subjectivity Score:* {subjectivity:.2f}")

        else:  # VADER
            scores = vader_analyzer.polarity_scores(text_input)
            compound = scores['compound']

            if compound >= 0.05:
                sentiment = "Positive 😊"
                color = "green"
            elif compound <= -0.05:
                sentiment = "Negative 😞"
                color = "red"
            else:
                sentiment = "Neutral 😐"
                color = "gray"

            st.markdown(f"*Sentiment:* <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
            st.write("*VADER Scores:*")
            st.json(scores)
