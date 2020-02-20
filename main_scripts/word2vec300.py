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

# import word2vec model
import gensim

def classification(x_train, y_train, x_test, y_test, clf):
    # train the model using the training sets
    clf.fit(x_train, y_train)

    # making predictions on the validation set
    y_pred = clf.predict(x_test)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Model Accuracy, how often is the classifier correct?
    print('Classifier:', clf)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('Confusion_mat:', confusion_mat)


def word_embedding(model, data_txt, length_row):
#     length_row = data_txt.shape[0]
    matrix = np.zeros((length_row,300))
    label = [0] *length_row
    for i in range(length_row):
        each_row = data_txt.loc[i].str.split() #.loc[].values.tolist()
        for words in each_row:
            for j,word in enumerate(words):
                if j ==0:
                    if word == '__label__match':
                        label[i] = 1
                    continue

                # replace __ and _ with space
                word = word.replace('__',' ')
                word = word.replace('_', ' ')
                word = re.findall(r'\w+',word) # list of word ['has', 'skill', 'resume']

                # use Word2Vec to reppresent each word and then reprensent ['has', 'skill', 'resume']
                string_word = [0] * 300  # empty list
                for each_word in word:
                    try:   # if model[each_word] exist():
                        string_word += model[each_word]
                    except:
                        continue
                string_word = string_word / len(word)

                # average the string_word to each row matrix
                matrix[i,:] += string_word
            matrix[i,:] /= len(words)

    word2vec_df = pd.DataFrame(matrix)
    print(word2vec_df.shape)
    print(len(label))

    return word2vec_df, label


if __name__ == "__main__":

    # All the classifiers we used here
    lr= linear_model.LogisticRegression()
    rf = RandomForestClassifier()
    dt = tree.DecisionTreeClassifier()
    # xgb_clf = xgb.XGBClassifier()

    # load train,validate, and test data; which was split before
    print('test1')
    output_txt = pd.read_csv('../data/processed/output.txt', sep='\t')  # all of the data

    # word2vec model for 300 dimension
    model = gensim.models.KeyedVectors.load_word2vec_format('./enwiki_20180420_300d.txt',binary=False,limit=500000)

    # word2vec model for 100 dimension
#     model = gensim.models.KeyedVectors.load_word2vec_format('./enwiki_20180420_100d.txt',binary=False,limit=500000)

    X, y = word_embedding(model, output_txt,output_txt.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    classification(X_train, y_train, X_test, y_test, lr)
    classification(X_train, y_train, X_test, y_test, rf)
