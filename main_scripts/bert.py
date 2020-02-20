import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
import json
import codecs

import re
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model
from sklearn import tree
# import xgboost as xgb
from sklearn.svm import SVC


def get_data_label(data_txt):
    length_row = data_txt.shape[0]
    y_label = [0] *length_row
    for i in range(length_row):
        each_row_label = data_txt.loc[i].str.split().tolist()[0][0]
        if each_row_label == '__label__match':
            y_label[i] = 1
    return y_label

# feature extraction by BERT
def bert_embedding(data_txt, embedding_model):
    length_row = data_txt.shape[0]

    sentences =[]
    for i in range(length_row):
        each_row = data_txt.loc[i].str.split().tolist()[0][1:] #list
        string = ' '.join(each_row)    # 'r_m_j h_p_m m_h_a'
        sentences.append(string)  # ['r_m_j h_p_m m_h_a', 'r_m_j h_p_m m_h_a']

    sentence_embeddings = embedding_model.encode(sentences)

    sentences_bert_list = []
    for sentence, embedding in zip(sentences, sentence_embeddings):
        sentences_bert_list.append(embedding.tolist())

    x_data_bert = pd.DataFrame(sentences_bert_list) #df

    return x_data_bert

# train the model by different classifiers
# validate the model
def classification(x_train, y_train, x_test, y_test, clf):
    # train the model using the training sets
    clf.fit(x_train, y_train)

    # making predictions on the validation set
    y_pred = clf.predict(x_test)

    # Model Accuracy, how often is the classifier correct?
    print('Classifier:', clf)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    # All the classifiers we used here
    lr= linear_model.LogisticRegression()
    rf = RandomForestClassifier()
    dt = tree.DecisionTreeClassifier()
    # xgb_clf = xgb.XGBClassifier()

    # load train,validate, and test data; which was split before
    print('test1')
    output_txt = pd.read_csv('../data/processed/output.txt', sep='\t')  # all of the data

    # BERT model named 'bert-base-nli-stsb-mean-tokens'
    model_stsb = SentenceTransformer('bert-base-nli-stsb-mean-tokens')  # mean --> max
    x_model_stsb = bert_embedding(output_txt,model_stsb)
    y = get_data_label(output_txt)

#     # bert model named 'bert-base-nli-mean-tokens'
#     model = SentenceTransformer('bert-base-nli-mean-tokens')

    X_train, X_test, y_train, y_test = train_test_split(x_model_stsb, y, test_size=0.33, random_state=42)

    print('train_txt shape:',X_train.shape)

    classification(X_train, y_train, X_test, y_test, lr)
    classification(X_train, y_train, X_test, y_test, rf)
    classification(X_train, y_train, X_test, y_test, dt)
    classification(X_train, y_train, X_test, y_test, xgb_clf)
