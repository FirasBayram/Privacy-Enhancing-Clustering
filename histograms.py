import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Read the dataset
data = pd.read_csv('dataframe.csv')
print(data.columns)

# Correlation Matrix
corr = data.corr()
fig, ax = plt.subplots(figsize=(20, 20))
cmap = cm.get_cmap('OrRd',15)
cax = ax.matshow(corr, cmap=cmap, vmin=-1, vmax=1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
plt.show()

#   Histogram of Age
data.Age.hist(bins=20)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#   Histogram of Gender
data.Gender.hist(bins=2)
plt.title('Histogram of Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

#   Histogram of Distance From Home
data.DistanceFromHome.hist(bins=20)
plt.title('Histogram of Distance From Home')
plt.xlabel('Distance From Home')
plt.ylabel('Frequency')
plt.show()

#   Histogram of Distance From Years At Company
data.YearsAtCompany.hist(bins=20)
plt.title('Histogram of Years At Company')
plt.xlabel('Years At Company')
plt.ylabel('Frequency')
plt.show()

#   Remove Outliers from Years At Company column
Q1 = data['YearsAtCompany'].quantile(0.25)
Q3 = data['YearsAtCompany'].quantile(0.8)
IQR = Q3 - Q1
fil = data["YearsAtCompany"].between(Q1-IQR, Q3+IQR)
data = data[fil]

#   Plot the Years At Company column after outliers' removal
data['YearsAtCompany'].value_counts()
sns.countplot(x='YearsAtCompany', data=data)
plt.show()

#   Apply SMOTE to over-sample the dataset so we get balanced classes
os = SMOTE(random_state=3)
X, Y = os.fit_sample(data[['Age',  'Gender', 'DistanceFromHome', 'MonthlyIncome',
                           'PerformanceRating', 'NumCompaniesWorked']],
                     data['YearsAtCompany'])
print(Y.value_counts())
data = pd.concat([X, Y], axis=1)
print(len(data))


#   Add cluster column for the data
def cluster(data):
    value = data['YearsAtCompany'].values
    print(len(value))
    row = []
    for i in value:
        if i in range(0, 3):
            row.append(0)
        if i in range(3, 6):
            row.append(1)
        if i in range(6, 9):
            row.append(2)
        if i in range(9, 12):
            row.append(3)
        if i in range(12, 15):
            row.append(4)
        if i in range(15, 18):
            row.append(5)
    row_ser = pd.Series(row)
    data['Cluster'] = row_ser.values


#   Apply the function cluster to or data
cluster(data)

#   Save the clustered data in csv file
data.to_csv('clustered.csv', index=False)

