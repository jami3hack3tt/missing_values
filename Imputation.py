import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer

print("Importing data")
data=pd.read_csv('data.csv', low_memory=False)
print(data.head(5))

#Counting number of columns
len(data.columns)

print("\n Subsetting the data to the important stuff")
important_data = data.iloc[1:20001,92:102]
print(important_data.head(5))

print("\n Checking the amount of missing values we have in our important data subset.")
missing_values_count =important_data.isnull().sum()
print(missing_values_count)

#percentage of missing values on important data
total_cells = np.product(important_data.shape)
total_missing = missing_values_count.sum()
print("\n Percentage of missing values:") 
print(total_missing/total_cells * 100)

print("\n Now running KNN imputation to replace missing values")
#Imputation using KNN
imputer = KNNImputer(n_neighbors=2, weights="uniform")

#Running imputation
X = imputer.fit_transform(important_data)

#Creating a new dataframe with missing values replaced with imputated values. 
important_data_nmv = pd.DataFrame(X, columns=important_data.columns, index=important_data.index)

print("\n Re-checking missing values after running KNN imputation")
#Running the missing value counter again
missing_values_count_nmv = important_data_nmv.isnull().sum()
print(missing_values_count_nmv)

#Showcasing data with missing values removed.
print("\n Showcasing sample of missing values imputated with KNN")
print(important_data_nmv.head(10))