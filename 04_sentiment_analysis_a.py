import nltk
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), 'html.parser')
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), 'html.parser')
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 3]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


X = []
Y = []

for review in positive_reviews:
    X.append(' '.join(my_tokenizer(review.text)))
    Y.append(1)

for review in negative_reviews:
    X.append(' '.join(my_tokenizer(review.text)))
    Y.append(0)

df = pd.DataFrame(data={'data': X, 'target': Y})

countVec = CountVectorizer(encoding="ignore")
X = countVec.fit_transform(df['data'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Score TRAIN NB: ', model.score(X_train, Y_train))
print('Score TEST NB: ', model.score(X_test, Y_test))

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Score TRAIN AdaBoost: ', model.score(X_train, Y_train))
print('Score TEST AdaBoost: ', model.score(X_test, Y_test))

model = RandomForestClassifier()
model.fit(X_train, Y_train)
print('Score TRAIN RandomForest: ', model.score(X_train, Y_train))
print('Score TEST RandomForest: ', model.score(X_test, Y_test))

model = LogisticRegression()
model.fit(X_train, Y_train)
print('Score TRAIN LogisticRegression: ', model.score(X_train, Y_train))
print('Score TEST LogisticRegression: ', model.score(X_test, Y_test))
