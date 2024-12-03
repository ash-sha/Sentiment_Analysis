import pandas as pd
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk



def prepare_data(amazon_data_path,amazon_test_path,movie_data_path):
    amazon_data = pd.read_csv(amazon_data_path)
    amazon_test = pd.read_csv(amazon_test_path)
    movie_data = pd.read_csv(movie_data_path)

    amazon_data.loc[len(amazon_data)] = amazon_data.columns
    amazon_data.columns = ["polarity","title",'review']
    amazon_data["polarity"] = amazon_data["polarity"].astype(int)
    amazon_data["polarity"] = amazon_data["polarity"].replace({1:"Negative",2:"Positive"})
    amazon_data["datatype"] = "Train"
    amazon_data["reviewtype"] = "Product"
    amazon_data = amazon_data.drop("title", axis=1)

    amazon_test.loc[len(amazon_test)] = amazon_test.columns
    amazon_test.columns = ["polarity","title",'review']
    amazon_test["polarity"] = amazon_test["polarity"].astype(int)
    amazon_test["polarity"] = amazon_test["polarity"].replace({1:"Negative",2:"Positive"})
    amazon_test["datatype"] = "Test"
    amazon_test["reviewtype"] = "Product"
    amazon_test = amazon_test.drop("title", axis=1)

    movie_data = movie_data.drop("Unnamed: 0", axis=1)
    movie_data.columns = ["review","polarity"]
    movie_data["polarity"] = movie_data["polarity"].astype(str)
    movie_data["reviewtype"] = "Movies"

    # split them equally so the case balance is preserved in split
    tta, ta = train_test_split(movie_data, test_size=0.15, stratify=movie_data['polarity'], random_state=42)
    tta['datatype'] = 'Train'
    ta['datatype'] = 'Test'

    movie_data_stratified = pd.concat([tta, ta]).reset_index(drop=True)
    movie_data_stratified = movie_data_stratified[["polarity","review","datatype","reviewtype"]]

    data = pd.concat([amazon_data,amazon_test,movie_data_stratified]).reset_index(drop=True)
    print("Data Prepared..")
    #split data

    train_text = data[data['datatype'] == "Train"]["review"].tolist()
    train_labels = data[data['datatype'] == "Train"]['polarity'].tolist()

    test_text = data[data['datatype'] == "Test"]["review"].tolist()
    test_labels = data[data['datatype'] == "Test"]['polarity'].tolist()
    print("Train, Test Splitted..")
    return data, train_text, test_text, train_labels, test_labels


# Initialize the lemmatizer and stopword list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text_nltk(text_list):
    '''Function to preprocess a list of texts by cleaning, lemmatizing, and removing unnecessary elements using NLTK.'''
    processed_texts = []

    for text in text_list:
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [
            lemmatizer.lemmatize(token) for token in tokens
            if token not in stop_words  # Remove stopwords
               and token not in string.punctuation  # Remove punctuation
               and token.isalpha()  # Keep only alphabetic words (no digits or symbols)
        ]

        processed_texts.append(" ".join(tokens))

    return processed_texts


