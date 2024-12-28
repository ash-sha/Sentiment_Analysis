## Sentiment Analysis on Amazon Review Dataset using MLOps  

This project showcases an end-to-end MLOps pipeline for sentiment analysis using the Amazon Review dataset.  

### Dataset Overview  

**Source**: [Amazon Review Dataset](https://jmcauley.ucsd.edu/data/amazon/)  
**Version**: 3 (Updated 09/09/2015)  

#### Origin  
The dataset contains Amazon reviews spanning 18 years, including ~35 million reviews up to March 2013. Each review includes product information, user details, ratings, and plaintext review content.  

For more details, refer to:  
- [J. McAuley and J. Leskovec, RecSys 2013](https://jmcauley.ucsd.edu/data/amazon/): *Hidden factors and hidden topics: understanding rating dimensions with review text*.  

The polarity dataset for this project is constructed by Xiang Zhang as a benchmark for text classification. For more details, refer to:  
- Xiang Zhang, Junbo Zhao, Yann LeCun, NIPS 2015: *Character-level Convolutional Networks for Text Classification*.  

#### Dataset Details  
- Reviews are classified as **positive** (scores 4 & 5) or **negative** (scores 1 & 2). Neutral reviews (score 3) are excluded.  
- Each class contains:  
  - 1,800,000 training samples  
  - 200,000 testing samples  

**Files**:  
- `train.csv` and `test.csv` (comma-separated values with the following columns):  
  - Class index (1 = Negative, 2 = Positive)  
  - Review title  
  - Review text  

**Special Formatting**:  
- Review title and text are enclosed in double quotes (`"`), and internal double quotes are escaped (`""`).  
- Newlines are represented as `\n`.  

### Deployment  

- **Docker Image**: `aswaths/sentimentanalysis`  
- **Streamlit App**: [Streamlit Sentiment Analysis App](https://sentimentanalysisnpl.streamlit.app)  

### Instructions  

1. **Prepare the Dataset**  
   - Download the dataset and place it in the `data` folder.  

2. **Run the FastAPI Server**  
   - Execute `serve.py` to launch the FastAPI server.  
   - Access API documentation at `http://localhost:8000/docs`.  

3. **Make Predictions**  
   - Use the following `curl` command to fetch predictions:  
     ```bash
     curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"query": "This is a great product!"}'
     ```  

4. **Build and Run Docker Image**  
   - To build the Docker image:  
     ```bash
     docker build -t <image_name> .
     docker run <image_name>
     ```  
   - Modify the `Dockerfile` as needed.  

5. **Push to DockerHub**  
   - Create a DockerHub repository, then upload your image:  
     ```bash
     docker login
     docker tag <image_name> <username>/<repo>:<tag>
     docker push <username>/<repo>:<tag>
     ```  

6. **Monitor Models with MLflow**  
   - Access the MLflow UI to track and monitor model performance.  
