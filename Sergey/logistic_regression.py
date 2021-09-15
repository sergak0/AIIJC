from typing import List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import re
import os
import pickle

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
from nltk.stem import WordNetLemmatizer 


class LogicticRegressionPreparator():
    def __init__(self):
        pass
    
    def data_prepare(self, language_dict, ru=False, en=False, es=False):
        if ru:
            stop_words = set(stopwords.words('russian'))
            lemmatizer = pymorphy2.MorphAnalyzer()
        elif en:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
        elif es:
            stop_words = set(stopwords.words('spanish'))
            lemmatizer = nltk.stem.SnowballStemmer('spanish')

        dict_prepared = []

        for  text in tqdm(language_dict):
            text = re.sub(r'[()]', '#', text)
            text = re.sub(r'#[0-9]+#','', text)
            text = re.sub(r'[^\w\s.]','', text.lower())

            word_tokens = word_tokenize(text)
            word_tokens = [w for w in word_tokens if not w in stop_words]
            if ru:
                word_tokens = [lemmatizer.parse(w)[0].normal_form for w in word_tokens]
            elif en:
                word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
            elif es:
                word_tokens = [lemmatizer.stem(w) for w in word_tokens]

            word_tokens = [w for w in word_tokens if not w in stop_words]

            filtered_text = ' '.join(word_tokens)
            filtered_text = re.sub(r' [.]+ ', ' ', filtered_text)
            dict_prepared.append(filtered_text)
        return dict_prepared
    
    def prepare_texts(self, texts):
        return self.data_prepare(texts, ru = True)


class LogicticRegressionModel():
  
  def __init__(self,
               pretrained = True,
               path: str = 'logistic_regression.pt'
              ):
    
    vect_path = os.path.join(path, 'vect.pickle')
    clf_path = os.path.join(path, 'clf.pickle')
    
    if not pretrained:
        self.vect = TfidfVectorizer()
        self.clf = LogisticRegression(penalty='l1', C=50 , verbose = 10 , solver='liblinear')
    else:
        self.vect = pickle.load(open(vect_path, 'rb'))
        self.clf = pickle.load(open(clf_path, 'rb'))

    self.model = Pipeline([('vect', self.vect),
                  ('clf', self.clf),
                  ])

    self.classes = ['животные', "музыка", "спорт", "литература"]

    
  def fit(self, X_train, Y_train):
    self.model.fit(X_train, Y_train)
    self.classes = self.model['clf'].classes_
  
  def predict(self, X_test: List[str]):
    return self.model.predict(X_test)
  
  def predict_proba(self, X_test: List[str]):
    return self.model.predict_proba(X_test)

  def show_metrics(self, X_test, Y_test):
    y_pred = self.predict(X_test)
    print('accuracy %s' % accuracy_score(y_pred, Y_test))
    print(classification_report(Y_test, y_pred, target_names=self.classes))

  def score(self, X_test, Y_test):
    return self.model.score(X_test, Y_test)


  def save(self, path: str = 'logistic_regression.pt'):   
    
    if not os.path.exists(path):
        os.makedirs(path)
     
    vect_path = os.path.join(path, 'vect.pickle')
    clf_path = os.path.join(path, 'clf.pickle')
    
    pickle.dump(self.vect, open(vect_path, 'wb'))
    pickle.dump(self.clf, open(clf_path, 'wb'))



if __name__ == '__main__':
  preparator = LogicticRegressionPreparator()

  data = pd.read_csv('marked_test.csv')
  del data['Unnamed: 0']

  nb_model = LogicticRegressionModel(path='logistic_regression.pt')

  X_test = preparator.prepare_texts(data['task'].values)
  
  data['prediction'] = nb_model.predict(X_test)
  data['score'] = nb_model.predict_proba(X_test).max(axis=1)

  data['prediction'] = [nb_model.classes[x] for x in data['prediction']]
  print(data.head())
