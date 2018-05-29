from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

data = pd.read_csv('spam.csv', encoding='iso-8859-1')
data = data.sample(frac=1)

X_brutu = data['v2']
Y_brutu = data['v1']

tfidfVect = TfidfVectorizer()
countVec = CountVectorizer(encoding="ignore")

tfidfVect.fit(X_brutu)
countVec.fit(X_brutu)

X_data_tfidf = tfidfVect.transform(X_brutu)
X_data_count = countVec.transform(X_brutu)

X_data_tfidf = X_data_tfidf.todense()
X_data_count = X_data_count.todense()
Y_data = (Y_brutu == 'spam').values.astype(int)

number_test = 1000

X_train = X_data_tfidf[:-number_test, :]
Y_train = Y_data[:-number_test]
X_test = X_data_tfidf[-number_test:, :]
Y_test = Y_data[-number_test:]

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Score  NB: ', model.score(X_test, Y_test))

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Score  AdaBoost: ', model.score(X_test, Y_test))

X_train = X_data_count[:-number_test, :]
X_test = X_data_count[-number_test:, :]

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Score  NB: ', model.score(X_test, Y_test))

model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Score  AdaBoost: ', model.score(X_test, Y_test))


