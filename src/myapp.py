import requests
import streamlit as st
from server import port_no
import subprocess
server_process = subprocess.Popen(["python", "server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
from pathlib import Path
import atexit


# Add a logo in the sidebar
logo_path = Path(__file__).parent / "pic.jpeg"
st.sidebar.image(str(logo_path), use_container_width=True)

st.title("Sentiment Analysis Tool")
st.write("""
##### Analyze the sentiment of your text in real-time.
""")

API_URL = f"http://localhost:{port_no}/predict"

query = st.text_input("Enter your Sentence:")

if st.button("Analyze"):
    if query:
        response = requests.post(API_URL, json = {"query": query})
        if response.status_code == 200:
            result = response.text
            output_box = st.text_area("Sentiment",result)
        else:
            st.error("Error calling API")
    else:
        st.warning("Please enter your query")


atexit.register(lambda: server_process.terminate())
