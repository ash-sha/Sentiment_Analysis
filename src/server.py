from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse  # Import the PlainTextResponse class
import uvicorn
from inference import predict_polarity

# Initialize FastAPI
app = FastAPI(title="Sentiment Analysis API", description="API for sentiment analysis", version="1.0")

# Define request model (only need query)
class SentimentRequest(BaseModel):
    query: str  # Enforce query length between 1 and 512 characters

# Define response model
class SentimentResponse(BaseModel):
    sentiment: str

# Basic endpoint to check if the server is working
@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running. Use /predict to analyze sentiment."}

# Define a route for sentiment analysis
@app.post("/predict", response_class=PlainTextResponse)
def predict_sentiment_api(request: SentimentRequest):
    try:
        # Call the inference function directly
        sentiment = predict_polarity(request.query)
        return sentiment # return sentiment directly as text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("access: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000,log_level='info')