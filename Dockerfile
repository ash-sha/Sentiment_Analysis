# Use a slim Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project files into the container
COPY . /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('wordnet')"

# Set environment variables for file paths
ENV AMAZON_DATA_PATH=/app/data/test_amazon.csv
ENV AMAZON_TEST_PATH=/app/data/test_amazon.csv
ENV MOVIE_DATA_PATH=/app/data/train.csv
ENV MODEL_SAVE_PATH=/app/models/sample_trained_model.pickle


# Expose the FastAPI app's port
EXPOSE 8000 8501

# Set the entry point to run the server.py inside the src folder
ENTRYPOINT ["streamlit", "run", "src/myapp.py"]
