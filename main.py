import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


data_df = pd.read_csv("./myData/spam.csv")
print(data_df.head(15))
print(data_df.info()) #understanding data set

print("\n")
for item in data_df.loc[1]:
    print(item)

# with pd.option_context("display.max_rows", 5572):
#     display(data_df)

#Utilizing CountVectorizer()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_df["Message"])
print(X)


#spliting the data
y = data_df["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("data from X_train")
print(X_train)
print("data from X_test")
print(X_test)


#training the model