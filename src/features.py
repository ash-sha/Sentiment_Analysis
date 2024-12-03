from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization

def feature(train_text_processed,test_text_processed,max_feature_num):
    vectorizer = TfidfVectorizer(max_features=max_feature_num)

    # Fit and transform training data, and transform test data
    train_vec = vectorizer.fit_transform(train_text_processed)
    test_vec = vectorizer.transform(test_text_processed)

    # Check the shape of train_vec to confirm it's 2D
    print("Shape of train_vec:", train_vec.shape)
    print("Shape of test_vec:", test_vec.shape)

    return train_vec, test_vec, vectorizer