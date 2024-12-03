# Use a base image with Python pre-installed
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project to the working directory
COPY . /app

# Install any dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the FastAPI app will use
EXPOSE 8000

# Set environment variables for paths
ENV AMAZON_DATA_PATH=/app/data/test_amazon.csv
ENV AMAZON_TEST_PATH=/app/data/test_amazon.csv
ENV MOVIE_DATA_PATH=/app/data/train.csv
ENV MODEL_SAVE_PATH=/app/models/sample_trained_model.pickle

# Command to run the FastAPI app
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
