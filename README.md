Amazon Review Polarity Dataset

source: https://jmcauley.ucsd.edu/data/amazon/

Version 3, Updated 09/09/2015

ORIGIN:

The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years,
 including ~35 million reviews up to March 2013. Reviews include product and user information, ratings,
 and a plaintext review. For more information, please refer to the following paper: J. McAuley and J. Leskovec.
 Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.

The Amazon reviews polarity dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from
 the above dataset. It is used as a text classification benchmark in the following paper:
  Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for
  Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).


DESCRIPTION:

The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as
negative, and 4 and 5 as positive. Samples of score 3 is ignored. In the dataset, class 1
is the negative and class 2 is the positive. Each class has 1,800,000 training samples and
200,000 testing samples.

The files train.csv and test.csv contain all the training samples as comma-sparated values.
There are 3 columns in them, corresponding to class index (1 or 2), review title and review
text. The review title and text are escaped using double quotes ("), and any internal double
quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed
with an "n" character, that is "\n".

docker image: aswaths/sentimentanalysis
 

INSTRUCTIONS:

1. load the data from specified source and drop into data folder
2. run serve.py to run fastAPI and access from localhost port 8000
3. localhost:8000/docs proves the API docs
4. in new terminal write below command to fetch the prediction. change the query to your desired query

        curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"query": "This is a great product!"}'
5. Additionally, modify the dockerfile if need or run below in terminal,
       
       docker build -t image_name .
       docker run image_name
6. Can push the code the docker hub if needed by creating a repository in DockerHub then in terminal run below to upload to DockerHub
       
       docker login
       docker tag image_name
       docker push username/repo:tag

7. Access **mlflow ui** to track and monitor models.