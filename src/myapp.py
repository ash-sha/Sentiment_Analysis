import requests
import streamlit as st
import subprocess
import time
import atexit
from pathlib import Path

# Start the server process
server_process = subprocess.Popen(["python", "server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Function to check if the server is up
def check_server_ready(url, retries=5, delay=2):
    for _ in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(delay)  # Wait before retrying
    return False

# Wait until the server is ready
if not check_server_ready("http://localhost:8000/predict"):
    st.error("Backend server is not ready yet.")
else:
    # Add a logo in the sidebar
    logo_path = Path(__file__).parent / "pic.jpeg"
    st.sidebar.image(str(logo_path), use_container_width=True)

    st.title("Sentiment Analysis Tool")
    st.write("""
    ##### Analyze the sentiment of your text in real-time.
    """)

    API_URL = "http://localhost:8005/predict"
    query = st.text_input("Enter your Sentence:")

    if st.button("Analyze"):
        if query:
            try:
                response = requests.post(API_URL, json={"query": query})
                if response.status_code == 200:
                    result = response.text
                    st.write("Model runs on Multinomial Naive Bayes, trained on 4 Million Amazon Product and Movie reviews with 78% accuracy")
                    st.text_area("Sentiment", result)
                else:
                    st.error(f"Error calling API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error connecting to the API: {str(e)}")
        else:
            st.warning("Please enter your query")

# Register the termination of the server process
atexit.register(lambda: server_process.terminate())
