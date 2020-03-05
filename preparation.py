import pandas as pd

#   Load the dataset
df = pd.read_csv('dataset.csv')

#   Convert Categorical Values into Numerical
attrition_map = {'Yes': 1, 'No': 0}
df = df.replace({'Attrition': attrition_map})
BT_map = {'Travel_Frequently': 2, 'Travel_Rarely': 1,  'Non-Travel': 0}
df = df.replace({'BusinessTravel': BT_map})
DP_map = {'Sales': 2, 'Research & Development': 1, 'Human Resources': 0}
df = df.replace({'Department': DP_map})
GD_map = {'Male': 2, 'Female': 1}
df = df.replace({'Gender': GD_map})
MS_map = {'Single': 2, 'Married': 1, 'Divorced': 0}
df = df.replace({'MaritalStatus': MS_map})

#   Print the type of each column's values
print(df.dtypes)

#   Save the dataframe in CSV file
df.to_csv('dataframe.csv', index=False)
