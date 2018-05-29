from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv('spam.csv', encoding='iso-8859-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ['labels', 'data']
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].as_matrix()

countVec = CountVectorizer(encoding="ignore")
X = countVec.fit_transform(df['data'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Score TEST  NB: ', model.score(X_test, Y_test))

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Score TEST  AdaBoost: ', model.score(X_test, Y_test))

tfidfVect = TfidfVectorizer(encoding="ignore")
X = tfidfVect.fit_transform(df['data'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Score TEST  NB: ', model.score(X_test, Y_test))

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Score TEST  AdaBoost: ', model.score(X_test, Y_test))
