import pandas as pd
import numpy as np
data = pd.read_csv("./myData/spam.csv")

print(data.shape)
print(data.head())

print("\n")
print(f"the type of data is {type(data)}")

#iterate throught the rows of data and print using iterrows() attribute for a datafrma object
for index, row in data.iterrows():
    # print(index)#from our result we have 5571 lines of code
    if index < 10:
        print(f"The index of the line is {index}, the category of this line is {row['Category']}")#the message works and prints only the category