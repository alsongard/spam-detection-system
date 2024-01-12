import pandas as pd

#create a DataFrame
data = {"Name": ["Alice","Bob", "Charlie"],
        "Age": [25, 30, 235],
        "City": ["New York", "San Francisco", "Los Angeles"]}#dictionary
print(data["Name"])
print(data["Age"])
print(data["City"][1])

print("\n")
df = pd.DataFrame(data)
print(df)
print(df.Name) # one can select data through the selecting the column
print("\n")

#iterate through the rows using iterrows() attribute for dataframes
for index,row in df.iterrows():
    print(index,row)
    print(f"The index is {index} and name is {row['Name']} and age is {row['Age']} and City is {row['City']}")
# it seems that the index represent the number of line the data is and the row represents the row of that index