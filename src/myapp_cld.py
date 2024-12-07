import requests
import streamlit as st
from pathlib import Path
import time

# Add a logo in the sidebar
logo_path = Path(__file__).parent / "pic.jpeg"
st.sidebar.image(str(logo_path), use_container_width=True)

st.title("Sentiment Analysis Tool")
st.write("""
##### Analyze the sentiment of your text in real-time.
""")

API_URL = "https://sentiment-analysis-3m74.onrender.com/predict"


# Cache the result of the sentiment analysis
@st.cache_data
def get_sentiment(query):
    response = requests.post(API_URL, json={"query": query})
    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Error calling API")


query = st.text_input("Enter your Sentence:")

if st.button("Analyze"):
    if query:
        with st.spinner("Analyzing..."):
            try:
                result = get_sentiment(query)
                st.write(
                    "Model runs on Multinomial Naive Bayes, trained on 4 Million Amazon Product and Movie reviews with 78% accuracy")

                # Display sentiment result
                if result.lower() == 'positive':
                    st.text_area("Sentiment", result, height=100, max_chars=500, key="positive", disabled=True)
                    st.markdown("<h3 style='color: green;'>Positive Sentiment</h3>", unsafe_allow_html=True)
                elif result.lower() == 'negative':
                    st.text_area("Sentiment", result, height=100, max_chars=500, key="negative", disabled=True)
                    st.markdown("<h3 style='color: red;'>Negative Sentiment</h3>", unsafe_allow_html=True)
                else:
                    st.text_area("Sentiment", result, height=100, max_chars=500, key="neutral", disabled=True)
                    st.markdown("<h3 style='color: gray;'>Neutral Sentiment</h3>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter your query")
