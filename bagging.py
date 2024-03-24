from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np


def bagging(embeddings, labels: list):

    np_bert_embeddings = embeddings.numpy()
    bagging_clf = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0).fit(np_bert_embeddings, labels)
