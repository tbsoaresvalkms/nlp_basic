import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

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


word_index_map = {}

positive_tokenized = []
negative_tokenized = []


def indexer_words(reviews, tokenized):
    for review in reviews:
        tokens = my_tokenizer(review.text)
        tokenized.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = len(word_index_map)


indexer_words(positive_reviews, positive_tokenized)
indexer_words(negative_reviews, negative_tokenized)


def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()
    x[-1] = label
    return x


N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))
i = 0

for tokens in positive_tokenized:
    data[i] = tokens_to_vector(tokens, 1)
    i += 1

for tokens in negative_tokenized:
    data[i] = tokens_to_vector(tokens, 0)
    i += 1

np.random.shuffle(data)

X = data[:, :-1]
Y = data[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

model = MultinomialNB()
model.fit(X_train, Y_train)
print('Score TRAIN NB: ', model.score(X_train, Y_train))
print('Score TEST NB: ', model.score(X_test, Y_test))

"""
model = AdaBoostClassifier()
model.fit(X_train, Y_train)
print('Score TRAIN AdaBoost: ', model.score(X_train, Y_train))
print('Score TEST AdaBoost: ', model.score(X_test, Y_test))
"""

model = RandomForestClassifier()
model.fit(X_train, Y_train)
print('Score TRAIN RandomForest: ', model.score(X_train, Y_train))
print('Score TEST RandomForest: ', model.score(X_test, Y_test))

model = LogisticRegression()
model.fit(X_train, Y_train)
print('Score TRAIN LogisticRegression: ', model.score(X_train, Y_train))
print('Score TEST LogisticRegression: ', model.score(X_test, Y_test))

threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
