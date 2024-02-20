import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("./myData/spam.csv")
df = pd.DataFrame(data)
print(df)
print(df.columns)
#convert it into numbers for conversion
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Message"])
print(X)

# #split data 2 train&test
y = df["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

clf = MultinomialNB()
clf.fit(X_train, y_train)