from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
Y = data[:, -1]

X_train = X[:-100, ]
Y_train = Y[:-100, ]
X_test = X[-100:, ]
Y_test = Y[-100:, ]

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Score  NB: ', model.score(X_test, Y_test))


model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Score  AdaBoost: ', model.score(X_test, Y_test))
