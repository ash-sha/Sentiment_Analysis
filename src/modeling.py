from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB


def modeling(train_vec,train_labels,test_vec,test_labels):

    clf = MultinomialNB().fit(train_vec, train_labels)

    # Predict on training data
    train_pred = clf.predict(train_vec)

    # Predict on test data
    test_pred = clf.predict(test_vec)

    # Accuracy
    train_accuracy = accuracy_score(train_labels, train_pred)
    test_accuracy = accuracy_score(test_labels, test_pred)

    # Classification Report (Precision, Recall, F1-Score)
    train_classification_report = classification_report(train_labels, train_pred)
    test_classification_report = classification_report(test_labels, test_pred)

    # Confusion Matrix
    train_confusion_matrix = confusion_matrix(train_labels, train_pred)
    test_confusion_matrix = confusion_matrix(test_labels, test_pred)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    print("\nTraining Classification Report:")
    print(train_classification_report)

    print("\nTest Classification Report:")
    print(test_classification_report)

    print("\nTraining Confusion Matrix:")
    print(train_confusion_matrix)

    print("\nTest Confusion Matrix:")
    print(test_confusion_matrix)

    return clf
