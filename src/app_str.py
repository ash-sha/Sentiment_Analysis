import os
import pickle
import streamlit as st
from pathlib import Path
import nltk

from data_processing import preprocess_text_nltk

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Add a logo in the sidebar
logo_path = Path(__file__).parent / "pic.jpeg"
st.sidebar.image(str(logo_path), use_container_width=True)

st.title("Sentiment Analysis Tool")
st.write("""
##### Analyze the sentiment of your text in real-time.
""")
# Function to load the model and vectorizer, and make predictions
def predict_polarity(user_query: str) -> str:
    import os
   
    model_path = os.path.join(os.getcwd(), 'models', 'sample_trained_model.pickle')
    saved_model_dic = pickle.load(open(model_path, "rb"))
    saved_clf = saved_model_dic['model']
    saved_vectorizer = saved_model_dic['vectorizer']

    # Preprocess the user query
    preprocessed_query = preprocess_text_nltk([user_query])

    # Transform the query text using the saved vectorizer
    query_vec = saved_vectorizer.transform(preprocessed_query)

    # Predict the polarity
    prediction = saved_clf.predict(query_vec)

    return prediction[0]
query = st.text_input("Enter your Sentence:")

if st.button("Analyze"):
    if query:
        result = predict_polarity(query)
        st.write("Model runs on Multinomial Naive Bayes, trained on 4 Million Amazon Product and Movie reviews with 78% accuracy")
        output_box = st.text_area("Sentiment",result)

    else:
        st.warning("Please enter your query")

