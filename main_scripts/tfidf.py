
import pandas as pd
import nltk
# import html2text # No module named 'html2text'
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import json
import codecs



def tf_idf(data_txt, length_row):
    
    total_list =[]
    for i in range(length_row):
        each_row = train_txt.loc[i].str.split().tolist()[0][1:] #list
        string = ' '.join(each_row)
        total_list.append(string)

    # print(total_list)

    vectorizer = TfidfVectorizer()
    train_tfidf_df = vectorizer.fit_transform(total_list)
    tfidf_keywords = vectorizer.get_feature_names()
    print(train_tfidf_df.shape)
    
    x_tfidf_matrix = pd.DataFrame(train_tfidf_df.toarray())
    x_tfidf_matrix.columns = tfidf_keywords    
    
    return x_tfidf_matrix, tfidf_keywords


if __name__ == "__main__":
    
    train_txt = pd.read_csv('data/ml/fasttext.golden.train.txt', sep='\t')
    validate_txt = pd.read_csv('data/ml/fasttext.golden.validation.txt', sep='\t')
    test_txt = pd.read_csv('data/ml/fasttext.golden.test.txt', sep='\t')

    model_vec50 = pd.read_csv('learning/data/models/resumemodel.50.vec', header = None, skiprows =1, sep=' ')
    model_vec50.drop(columns=[101],axis=1,inplace=True)    
    
    x_tfidf_matrix, tfidf_keywords = tf_idf(train_txt,train_txt.shape[0])


    ft_dict_filtered = model_vec50[model_vec50[0].isin(tfidf_keywords)]
    x_tfidf_matrix_final = x_tfidf_matrix[ft_dict_filtered[0].tolist()]

    X_train_tfidf = np.dot(x_tfidf_matrix_final, ft_dict_filtered)

    print(type(X_train_tfidf))
