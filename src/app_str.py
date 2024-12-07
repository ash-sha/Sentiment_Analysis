
import streamlit as st
from pathlib import Path
from inference import predict_polarity
import nltk
nltk.download('stopwords')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', '/Users/aswathshakthi/PycharmProjects/MLOps/Sentiment_Analysis/models/sample_trained_model.pickle')

# Add a logo in the sidebar
logo_path = Path(__file__).parent / "pic.jpeg"
st.sidebar.image(str(logo_path), use_container_width=True)

st.title("Sentiment Analysis Tool")
st.write("""
##### Analyze the sentiment of your text in real-time.
""")


query = st.text_input("Enter your Sentence:")

if st.button("Analyze"):
    if query:

        result = predict_polarity(query)
        st.write("Model runs on Multinomial Naive Bayes, trained on 4 Million Amazon Product and Movie reviews with 78% accuracy")
        output_box = st.text_area("Sentiment",result)

    else:
        st.warning("Please enter your query")

