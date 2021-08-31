import random
import re
import string
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import warnings 
from sklearn.linear_model import LogisticRegression
from PIL import Image
from IPython.display import clear_output
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import List
from sklearn.metrics import classification_report
import pickle
from typing import List
import nltk
import string
import pymorphy2
import codecs

warnings.filterwarnings(action='ignore',category=UserWarning, module='gensim')  
warnings.filterwarnings(action='ignore',category=FutureWarning, module='gensim')  
warnings.filterwarnings(action='ignore',category=DeprecationWarning, module='gensim')
warnings.filterwarnings(action='ignore',category=DeprecationWarning, module='smart_open') 
warnings.filterwarnings(action='ignore',category=DeprecationWarning, module='sklearn')
warnings.filterwarnings(action='ignore',category=DeprecationWarning, module='scipy')    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Preparator():
  def __init__(self,
               stopwords_path: str = '/content/drive/MyDrive/aiijc_sber/all_stopwords.txt',
               bigram_model_path: str = '/content/drive/MyDrive/aiijc_sber/bigram_model.pkl'):
      self.morph = pymorphy2.MorphAnalyzer()
      self.tokenizer = nltk.WordPunctTokenizer()
      self.stopwords = set(line.strip() for line in codecs.open(stopwords_path, "r", "utf_8_sig").readlines())
      self.bigram_mod = gensim.models.Phrases.load(bigram_model_path)

  def prepare_corp(self, news_list: List[str]):
      return [self.newstext2token(news_text) for news_text in news_list]

  def newstext2token(self, news_text: str):
      tokens = self.tokenizer.tokenize(news_text.lower())
      tokens_with_no_punct = [self.morph.parse(w)[0].normal_form for w in tokens if all(c in 'йцукенгшщзхъёфывапролджэячсмитьбю' for c in w)]
      tokens_base_forms = [w for w in tokens_with_no_punct if w not in self.stopwords]
      tokens_last = [w for w in tokens_base_forms if len(w)>1]
      tokens_bigrammed = self.make_bigrams(tokens_last)
      return ' '.join(tokens_bigrammed)

  def make_bigrams(self, doc):
      return self.bigram_mod[doc]

  def prepare_texts(self, texts):
    return self.prepare_corp(texts)
    


class BayesModel():
  
  def __init__(self,
               pretrained = True,
               vect_path: str = '/content/drive/MyDrive/aiijc_sber/nb/nb_vect.sav',
               tfidf_path: str = '/content/drive/MyDrive/aiijc_sber/nb/nb_tfidf.sav',
               clf_path: str = '/content/drive/MyDrive/aiijc_sber/nb/nb_clf.sav'):

    if not pretrained:
        self.vect = CountVectorizer()
        self.tfidf = TfidfTransformer()
        self.clf = MultinomialNB()
    else:
        vect = pickle.load(open(vect_path, 'rb'))
        tfidf = pickle.load(open(tfidf_path, 'rb'))
        clf = pickle.load(open(clf_path, 'rb'))

    self.nb = Pipeline([('vect', vect),
                  ('tfidf', tfidf),
                  ('clf', clf),
                  ])

    self.classes = ['животные', "музыка", "спорт", "литература"]

  

  def fit(self, X_train, Y_train):
    self.nb.fit(X_train, Y_train)
  
  def predict(self, X_test: List[str]):
    return self.nb.predict(X_test)
  
  def predict_proba(self, X_test: List[str]):
    return self.nb.predict_proba(X_test)

  def show_metrics(self, X_test, Y_test):
    y_pred = self.predict(X_test)
    print('accuracy %s' % accuracy_score(y_pred, Y_test))
    print(classification_report(Y_test, y_pred, target_names=self.classes))

  def score(self, X_test, Y_test):
    return self.nb.score(X_test, Y_test)


  def save_models(self,
                  vect_path: str = '/content/drive/MyDrive/aiijc_sber/nb/nb_vect.sav',
                  tfidf_path: str = '/content/drive/MyDrive/aiijc_sber/nb/nb_tfidf.sav',
                  clf_path: str = '/content/drive/MyDrive/aiijc_sber/nb/nb_clf.sav'):
    pickle.dump(self.vect, open(vect_path, 'wb'))
    pickle.dump(self.tfidf, open(tfidf_path, 'wb'))
    pickle.dump(self.clf, open(clf_path, 'wb'))

    
if __name__ == '__main__':
	preparator = Preparator(stopwords_path='Lisa/all_stopwords.txt',
	                        bigram_model_path='Lisa/bigram_model.pkl')

	data = pd.read_csv('marked_test.csv')
	del data['Unnamed: 0']

	nb_model = BayesModel(vect_path='Lisa/nb_vect.sav', 
	                         tfidf_path='Lisa/nb_tfidf.sav',
	                         clf_path='Lisa/nb_clf.sav',
	                        )

	X_test = preparator.prepare_texts(data['task'].values)
	
	data['prediction'] = nb_model.predict(X_test)
	data['score'] = nb_model.predict_proba(X_test).max(axis=1)

	data['prediction'] = [nb_model.classes[x] for x in data['prediction']]
	print(data.head())