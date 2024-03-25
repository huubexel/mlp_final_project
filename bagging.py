from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier


def bagging(train_word_embeddings, train_labels, test_word_embeddings):
    """
    Makes the bagging classifier with the GradientBoosting classifier and predicts with that model.
    It returns the prediction.
    """

    # Get the bagging classifier to predict the labels with fitting on the training data set
    bagging_classifier = BaggingClassifier(estimator=GradientBoostingClassifier(),
                                           n_estimators=10, random_state=0).fit(train_word_embeddings, train_labels)
    # Return the predictions
    return bagging_classifier.predict(test_word_embeddings)
