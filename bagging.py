from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier


def bagging(train_word_embeddings, train_labels, test_word_embeddings):

    # Get the bagging classifier to predict the labels with
    bagging_classifier = BaggingClassifier(estimator=GradientBoostingClassifier(),
                                           n_estimators=10, random_state=0).fit(train_word_embeddings, train_labels)

    return bagging_classifier.predict(test_word_embeddings)
