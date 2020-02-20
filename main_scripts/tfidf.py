
import pandas as pd
import nltk
# import html2text # No module named 'html2text'
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import json
import codecs

def get_data_label(data_txt):
    length_row = data_txt.shape[0]
    y_label = [0] *length_row
    for i in range(length_row):
        each_row_label = data_txt.loc[i].str.split().tolist()[0][0]
        if each_row_label == '__label__match':
            y_label[i] = 1
    return y_label

def tf_idf(data_txt, length_row):

    total_list =[]
    for i in range(length_row):
        each_row = data_txt.loc[i].str.split().tolist()[0][1:] #list
        string = ' '.join(each_row)
        total_list.append(string)

    # print(total_list)

    vectorizer = TfidfVectorizer()
    output_tfidf_df = vectorizer.fit_transform(total_list)
    tfidf_keywords = vectorizer.get_feature_names()
    print(output_tfidf_df.shape)

    x_tfidf_matrix = pd.DataFrame(output_tfidf_df.toarray())
    x_tfidf_matrix.columns = tfidf_keywords

    return x_tfidf_matrix, tfidf_keywords

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


    x_tfidf_matrix, tfidf_keywords = tf_idf(output_txt,output_txt.shape[0])
    y = get_data_label(output_txt)

    X_train, X_test, y_train, y_test = train_test_split(x_tfidf_matrix, y, test_size=0.33, random_state=42)

    classification(X_train, y_train, X_test, y_test, lr)
    classification(X_train, y_train, X_test, y_test, rf)
