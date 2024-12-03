from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr
from fastapi.responses import PlainTextResponse  # Import the PlainTextResponse class
from inference import predict_polarity  # Import your inference function

# Initialize FastAPI
app = FastAPI(title="Sentiment Analysis API", description="API for sentiment analysis", version="1.0")

# Define request model (only need query)
class SentimentRequest(BaseModel):
    query: constr(min_length=1, max_length=512)  # Enforce query length between 1 and 512 characters

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


## RUN FastAPI server - uvicorn src.server:app --reload
## 127.0.0.1:8000/docs - for docs . this is how fastapi works.
### curl http://127.0.0.1:8000/  - gives a response
# use curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"query": "I love this Product"}' to infer / we can also use postman to check
