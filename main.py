import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
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
clf = MultinomialNB()
clf.fit(X_train, y_train)

#model evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#testing the model
message = vectorizer.transform(["Today's Offer!, Claim your 3000$! Text YES to 3034 now! To win your prize"])
prediction = clf.predict(message)
print("The email is :", prediction[0])