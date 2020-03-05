import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#   Load the dataset
data = pd.read_csv('clustered.csv')

#   Number of iterations to run for clustering
iter = 50

#   Number of Parameters to use for data distortions
dpar = 30


# Root mean square error function
def rmse(y_actual,y_predicted):
    errorsquare = (y_actual-y_predicted)**2
    sum_of_error_square = np.mean(errorsquare)
    rmse = np.sqrt(sum_of_error_square)
    return(rmse)


# Define Error Measurements
ds_rmse = 0     #RMSE of the dataset
me = 0     #Test Misclassification Error
wcss = 0        #Within-Cluster-Sum-of-Squares


#   TDP Technique
tdp = data.copy()
age_tdp = np.linspace(1, 10, dpar)
gender_tdp = np.linspace(1, 4, dpar)
disho_tdp = np.linspace(1, 5, dpar)
moninc_tdp = np.linspace(1, 50, dpar)
perf_tdp = np.linspace(1, 10, dpar)
tdp_rmse = np.zeros(dpar)       # TDP RMSE
tdp_me = np.zeros(dpar)         # TDP Test ME
tdp_var = np.zeros((dpar, 6))    # TDP Variance
tdp_wcss = np.zeros(dpar)       # TDP WCSS

#   SDP Technique
sdp = data.copy()
age_sdp = np.linspace(1, 1.5, dpar)
gender_sdp = np.linspace(1, 1.3, dpar)
disho_sdp = np.linspace(1, 1.1, dpar)
moninc_sdp = np.linspace(1, 1.4, dpar)
perf_sdp = np.linspace(1, 1.2, dpar)
sdp_rmse = np.zeros(dpar)       # SDP RMSE
sdp_me = np.zeros(dpar)         # SDP Test ME
sdp_var = np.zeros((dpar,6))    # SDP Variance
sdp_wcss = np.zeros(dpar)       # SDP WCSS

#   RDP Technique
rdp = data.copy()
angles = np.linspace(2, 75, dpar)
rdp_rmse = np.zeros(dpar)       # RDP RMSE
rdp_me = np.zeros(dpar)         # RDP Test ME
rdp_var = np.zeros((dpar,6))    # RDP Variance
rdp_wcss = np.zeros(dpar)       # RDP WCSS

# Enhanced TDP
enh_tdp_rmse = np.zeros(dpar)
enh_tdp_me = np.zeros(dpar)
enh_tdp_var = np.zeros((dpar,6))
enh_tdp_wcss = np.zeros(dpar)

# Enhanced SDP
enh_sdp_rmse= np.zeros(dpar)
enh_sdp_me = np.zeros(dpar)
enh_sdp_var = np.zeros((dpar, 6))
enh_sdp_wcss = np.zeros(dpar)

# Enhanced RDP
enh_rdp_rmse = np.zeros(dpar)
enh_rdp_me = np.zeros(dpar)
enh_rdp_var = np.zeros((dpar, 6))
enh_rdp_wcss = np.zeros(dpar)

# Data Perturbation Probability
prob = np.linspace(0.05, 0.25, dpar)

print('start')
for r in range(0, iter):
    #   Split train and test dataset for clustering
    splt = np.random.rand(len(data)) < 0.75
    train_data = data[splt]
    test_data = data[~splt]

    #   Dataset clustering Before Transformation
    kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10).\
        fit(train_data.iloc[:, 0:-2].values)
    orig_pred = kmeans.predict(test_data.iloc[:, 0:-2].values)
    label_true = test_data.iloc[:, -1]
    ds_rmse = ds_rmse+rmse(label_true, orig_pred)
    wcss = wcss + kmeans.inertia_

    #   TDP Clustering
    #   Split train and test dataset
    train_translated = tdp[splt]
    test_translated = tdp[~splt]
    for i in range(0, dpar):
        #   Perform Translation Data Perturbation
        train_translated['Age'] = (data['Age'] + age_tdp[i])
        train_translated['Gender'] = (data['Gender'] + gender_tdp[i])
        train_translated['DistanceFromHome'] = (data['DistanceFromHome'] + disho_tdp[i])
        train_translated['MonthlyIncome'] = (data['MonthlyIncome'] + moninc_tdp[i])
        train_translated['PerformanceRating'] = (data['PerformanceRating'] + perf_tdp[i])
        test_translated['Age'] = (data['Age'] + age_tdp[i])
        test_translated['Gender'] = (data['Gender'] + gender_tdp[i])
        test_translated['DistanceFromHome'] = (data['DistanceFromHome'] + disho_tdp[i])
        test_translated['MonthlyIncome'] = (data['MonthlyIncome'] + moninc_tdp[i])
        test_translated['PerformanceRating'] = (data['PerformanceRating'] + perf_tdp[i])
        tdp[splt] = train_translated
        tdp[~splt] = test_translated
        train_translated.to_csv('tdp_tra.csv', index=False)
        #   Apply KMeans
        kmeans = KMeans(n_clusters=6, init='k-means++', n_init=20).\
            fit(train_translated.iloc[:, 0:-2].values)
        labels = kmeans.labels_
        #   Calculate Test Error
        y_pred = kmeans.predict(test_translated.iloc[:, 0:-2].values)
        tdp_rmse[i] = tdp_rmse[i] + rmse(orig_pred, y_pred)
        tdp_wcss[i] = tdp_wcss[i] + kmeans.inertia_
        test_error = 0
        for j in range(0, 6):
            test_error = test_error + abs(sum(orig_pred == j) - sum(y_pred == j))
        tdp_me[i] = tdp_me[i] + (test_error / len(y_pred))
        #   Calculate the variance for each attribute
        tdp_var[i, 0] = tdp_var[i, 0] + np.var(data['Age'] - tdp['Age']) / np.var(data['Age'])
        tdp_var[i, 1] = tdp_var[i, 1] + np.var(data['Gender'] - tdp['Gender']) / np.var(data['Gender'])
        tdp_var[i, 2] = tdp_var[i, 2] + np.var(data['DistanceFromHome'] - tdp['DistanceFromHome']) / np.var(data['DistanceFromHome'])
        tdp_var[i, 3] = tdp_var[i, 3] + np.var(data['MonthlyIncome'] - tdp['MonthlyIncome']) / np.var(data['MonthlyIncome'])
        tdp_var[i, 4] = tdp_var[i, 4] + np.var(data['PerformanceRating'] - tdp['PerformanceRating']) / np.var(data['PerformanceRating'])

    #   SDP Clustering
    #   Split train and test dataset
    train_scaled = sdp[splt]
    test_scaled = sdp[~splt]
    #   Perform Translation Data Perturbation
    for i in range(0, dpar):
        train_scaled['Age'] = (data['Age'] * age_sdp[i])
        train_scaled['Gender'] = (data['Gender'] * gender_sdp[i])
        train_scaled['DistanceFromHome'] = (data['DistanceFromHome'] * disho_sdp[i])
        train_scaled['MonthlyIncome'] = (data['MonthlyIncome'] * moninc_sdp[i])
        train_scaled['PerformanceRating'] = (data['PerformanceRating'] * perf_sdp[i])
        test_scaled['Age'] = (data['Age'] * age_sdp[i])
        test_scaled['Gender'] = (data['Gender'] * gender_sdp[i])
        test_scaled['DistanceFromHome'] = (data['DistanceFromHome'] * disho_sdp[i])
        test_scaled['MonthlyIncome'] = (data['MonthlyIncome'] * moninc_sdp[i])
        test_scaled['PerformanceRating'] = (data['PerformanceRating'] * perf_sdp[i])
        train_scaled.to_csv('sdp_tra.csv', index=False)
        sdp[splt] = train_scaled
        sdp[~splt] = test_scaled
        train_scaled.to_csv('sdp_tra.csv', index=False)
        #   Apply KMeans
        kmeans = KMeans(n_clusters=6, init='k-means++', n_init=20).fit(train_scaled.iloc[:, 0:-2].values)
        labels = kmeans.labels_
        #   Calculate Test Error
        sdp_wcss[i] = sdp_wcss[i] + kmeans.inertia_
        y_pred = kmeans.predict(test_scaled.iloc[:, 0:-2].values)
        sdp_rmse[i] = sdp_rmse[i] + rmse(orig_pred, y_pred)
        test_error = 0
        for j in range(0, 6):
            test_error = test_error + abs(sum(orig_pred == j) - sum(y_pred == j))
        #   Calculate the variance for each attribute
        sdp_me[i] = sdp_me[i] + (test_error / len(y_pred))
        sdp_var[i, 0] = sdp_var[i, 0] + np.var(data['Age'] - sdp['Age']) / np.var(data['Age'])
        sdp_var[i, 1] = sdp_var[i, 1] + np.var(data['Gender'] - sdp['Gender']) / np.var(data['Gender'])
        sdp_var[i, 2] = sdp_var[i, 2] + np.var(data['DistanceFromHome'] - sdp['DistanceFromHome']) / np.var(data['DistanceFromHome'])
        sdp_var[i, 3] = sdp_var[i, 3] + np.var(data['MonthlyIncome'] - sdp['MonthlyIncome']) / np.var(data['MonthlyIncome'])
        sdp_var[i, 4] = sdp_var[i, 4] + np.var(data['PerformanceRating'] - sdp['PerformanceRating']) / np.var(data['PerformanceRating'])

    #   RDP Clustering
    train_angle_rot = rdp[splt]
    test_angle_rot = rdp[~splt]
    for i in range(0, dpar):
        #   Angle used to perturb the data
        angle = np.radians(angles[i])
        #   Transformation matrix for Rotation
        c, s = np.cos(angle), np.sin(angle)
        TM = np.array(((c, -s), (s, c)))
        #   Define the angle we want to use to rotate the two cloumns
        angle1 = data[['Age', 'YearsAtCompany']]
        angle2 = data[['Gender', 'YearsAtCompany']]
        angle3 = data[['DistanceFromHome', 'YearsAtCompany']]
        angle4 = data[['MonthlyIncome', 'YearsAtCompany']]
        angle5 = data[['PerformanceRating', 'YearsAtCompany']]
        angle_rot1 = np.matmul(angle1, TM)
        angle_rot2 = np.matmul(angle2, TM)
        angle_rot3 = np.matmul(angle3, TM)
        angle_rot4 = np.matmul(angle4, TM)
        angle_rot5 = np.matmul(angle5, TM)
        #   The rotation array
        rotation_arr = np.c_[angle_rot1.iloc[:, [0]], angle_rot2.iloc[:, [0]], angle_rot3.iloc[:, [0]],
                             angle_rot4.iloc[:, [0]], angle_rot5.iloc[:, [0]]]
        rotation_arr[:, [0, 1, 2, 3, 4]] = rotation_arr[:,[0, 1, 2, 3, 4]]
        rdp[['Age', 'Gender', 'DistanceFromHome', 'MonthlyIncome', 'PerformanceRating']] = abs(rotation_arr)
        rdp.to_csv('rdp_tra.csv', index=False)
        train_angle_rot = rdp[splt]
        test_angle_rot = rdp[~splt]
        kmeans = KMeans(n_clusters = 6, init = 'k-means++',n_init=20).fit(train_angle_rot.iloc[:,0:-2].values)
        labels = kmeans.labels_
        y_pred = kmeans.predict(test_angle_rot.iloc[:, 0:-2].values)
        #   Calculate Test Error
        rdp_rmse[i] = rdp_rmse[i] + rmse(orig_pred, y_pred)
        rdp_wcss[i] = rdp_wcss[i]+kmeans.inertia_
        test_error = 0
        for j in range(0,6):
            test_error = test_error + abs(sum(orig_pred==j)-sum(y_pred==j))
        rdp_me[i] = rdp_me[i] + (test_error/len(y_pred))
        #   Calculate the variance for each attribute
        rdp_var[i,0] = rdp_var[i,0] + np.var(data['Age']-rdp['Age'])/ np.var(data['Age'])
        rdp_var[i,1] = rdp_var[i,1] + np.var(data['Gender']-rdp['Gender'])/ np.var(data['Gender'])
        rdp_var[i,2] = rdp_var[i,2] + np.var(data['DistanceFromHome']-rdp['DistanceFromHome'])/ np.var(data['DistanceFromHome'])
        rdp_var[i,3] = rdp_var[i,3] + np.var(data['MonthlyIncome']-rdp['MonthlyIncome'])/ np.var(data['MonthlyIncome'])
        rdp_var[i,4] = rdp_var[i,4] + np.var(data['PerformanceRating']-rdp['PerformanceRating'])/ np.var(data['PerformanceRating'])

    #   Improving Privacy
    #   TDP
    for i in range(0, dpar):
        #   Step 1: We select a probability distribution
        P = prob[i]
        #   Step 2: We randomly select P% of the vectors
        noise_p = int(3528 * P)
        #   Step 3: Based on the previous steps, we distort the selected vectors
        n_age = np.transpose(np.random.normal(loc=0, scale=10, size=(1, noise_p)))
        n_gen = np.transpose(np.random.normal(loc=0, scale=4, size=(1, noise_p)))
        n_dis = np.transpose(np.random.normal(loc=0, scale=5, size=(1, noise_p)))
        n_moninc = np.transpose(np.random.normal(loc=0, scale=50, size=(1, noise_p)))
        n_perrat = np.transpose(np.random.normal(loc=0, scale=10, size=(1, noise_p)))
        enhanced_tdp = tdp.copy()
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_tdp['Age'][perturbed_idx] = enhanced_tdp['Age'][perturbed_idx] + n_age[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_tdp['Gender'][perturbed_idx] = enhanced_tdp['Gender'][perturbed_idx] + n_gen[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_tdp['DistanceFromHome'][perturbed_idx] = enhanced_tdp['DistanceFromHome'][perturbed_idx] + n_dis[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_tdp['MonthlyIncome'][perturbed_idx] = enhanced_tdp['MonthlyIncome'][perturbed_idx] + n_moninc[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_tdp['PerformanceRating'][perturbed_idx] = enhanced_tdp['PerformanceRating'][perturbed_idx] + n_perrat[:, 0]
        train_translated = enhanced_tdp[splt]
        test_translated = enhanced_tdp[~splt]
        kmeans = KMeans(n_clusters=6, init='k-means++').fit(train_translated.iloc[:, 0:-2].values)
        labels = kmeans.labels_
        enh_tdp_wcss[i] = enh_tdp_wcss[i] + kmeans.inertia_
        y_pred = kmeans.predict(test_translated.iloc[:, 0:-2].values)
        enh_tdp_rmse[i] = enh_tdp_rmse[i] + rmse(orig_pred, y_pred)
        train_total = 0
        test_error = 0
        for j in range(0, 6):
            test_error = test_error + abs(sum(orig_pred == j) - sum(y_pred == j))
        enh_tdp_me[i] = enh_tdp_me[i] + (test_error / len(y_pred))
        enh_tdp_var[i, 0] = enh_tdp_var[i, 0] + np.var(data['Age'] - enhanced_tdp['Age']) / np.var(data['Age'])
        enh_tdp_var[i, 1] = enh_tdp_var[i, 1] + np.var(data['Gender'] - enhanced_tdp['Gender']) / np.var(data['Gender'])
        enh_tdp_var[i, 2] = enh_tdp_var[i, 2] + np.var(data['DistanceFromHome'] - enhanced_tdp['DistanceFromHome']) / np.var(data['DistanceFromHome'])
        enh_tdp_var[i, 3] = enh_tdp_var[i, 3] + np.var(data['MonthlyIncome'] - enhanced_tdp['MonthlyIncome']) / np.var(data['MonthlyIncome'])
        enh_tdp_var[i, 4] = enh_tdp_var[i, 4] + np.var(data['PerformanceRating'] - enhanced_tdp['PerformanceRating']) / np.var(data['PerformanceRating'])

    # SDP
    for i in range(0, dpar):
        P = prob[i]
        noise_p = int(3528 * P)
        n_age = np.transpose(np.random.normal(loc=0, scale=10, size=(1, noise_p)))
        n_gen = np.transpose(np.random.normal(loc=0, scale=4, size=(1, noise_p)))
        n_dis = np.transpose(np.random.normal(loc=0, scale=50, size=(1, noise_p)))
        n_moninc = np.transpose(np.random.normal(loc=0, scale=5, size=(1, noise_p)))
        n_perrat = np.transpose(np.random.normal(loc=0, scale=10, size=(1, noise_p)))
        enhanced_sdp = sdp.copy()
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_sdp['Age'][perturbed_idx] = enhanced_sdp['Age'][perturbed_idx] + n_age[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_sdp['Gender'][perturbed_idx] = enhanced_sdp['Gender'][perturbed_idx] + n_gen[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_sdp['DistanceFromHome'][perturbed_idx] = enhanced_sdp['DistanceFromHome'][perturbed_idx] + n_dis[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_sdp['MonthlyIncome'][perturbed_idx] = enhanced_sdp['MonthlyIncome'][perturbed_idx] + n_moninc[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_sdp['PerformanceRating'][perturbed_idx] = enhanced_sdp['PerformanceRating'][perturbed_idx] + n_perrat[:, 0]
        train_scaled = enhanced_sdp[splt]
        test_scaled = enhanced_sdp[~splt]
        kmeans = KMeans(n_clusters=6, init='k-means++').fit(train_scaled.iloc[:, 0:-2].values)
        labels = kmeans.labels_
        enh_sdp_wcss[i] = enh_sdp_wcss[i] + kmeans.inertia_
        y_pred = kmeans.predict(test_scaled.iloc[:, 0:-2].values)
        enh_sdp_rmse[i] = enh_sdp_rmse[i] + rmse(orig_pred, y_pred)
        test_error = 0
        for j in range(0, 6):
            test_error = test_error + abs(sum(orig_pred == j) - sum(y_pred == j))
        enh_sdp_me[i] = enh_sdp_me[i] + (test_error / len(y_pred))
        enh_sdp_var[i, 0] = enh_sdp_var[i, 0] + np.var(data['Age'] - enhanced_sdp['Age']) / np.var(data['Age'])
        enh_sdp_var[i, 1] = enh_sdp_var[i, 1] + np.var(data['Gender'] - enhanced_sdp['Gender']) / np.var(data['Gender'])
        enh_sdp_var[i, 2] = enh_sdp_var[i, 2] + np.var(data['DistanceFromHome'] -enhanced_sdp['DistanceFromHome']) / np.var(data['DistanceFromHome'])
        enh_sdp_var[i, 3] = enh_sdp_var[i, 3] + np.var(data['MonthlyIncome'] - enhanced_sdp['MonthlyIncome']) / np.var(data['MonthlyIncome'])
        enh_sdp_var[i, 4] = enh_sdp_var[i, 4] + np.var(data['PerformanceRating'] - enhanced_sdp['PerformanceRating']) / np.var(data['PerformanceRating'])

    # RDP
    for i in range(0, dpar):
        P = prob[i]
        noise_p = int(3528 * P)
        n_age = np.transpose(np.random.normal(loc=0, scale=10, size=(1, noise_p)))
        n_gen = np.transpose(np.random.normal(loc=0, scale=4, size=(1, noise_p)))
        n_dis = np.transpose(np.random.normal(loc=0, scale=50, size=(1, noise_p)))
        n_moninc = np.transpose(np.random.normal(loc=0, scale=5, size=(1, noise_p)))
        n_perrat = np.transpose(np.random.normal(loc=0, scale=10, size=(1, noise_p)))
        enhanced_rdp = rdp.copy()
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_rdp['Age'][perturbed_idx] = enhanced_rdp['Age'][perturbed_idx] + n_age[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_rdp['Gender'][perturbed_idx] = enhanced_rdp['Gender'][perturbed_idx] + n_gen[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_rdp['DistanceFromHome'][perturbed_idx] = enhanced_rdp['DistanceFromHome'][perturbed_idx] + n_dis[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_rdp['MonthlyIncome'][perturbed_idx] = enhanced_rdp['MonthlyIncome'][perturbed_idx] + n_moninc[:, 0]
        perturbed_idx = np.random.permutation(range(0, 3528))[0:noise_p]
        enhanced_rdp['PerformanceRating'][perturbed_idx] = enhanced_rdp['PerformanceRating'][perturbed_idx] + n_perrat[:, 0]
        train_angle_rot = enhanced_rdp[splt]
        test_angle_rot = enhanced_rdp[~splt]
        kmeans = KMeans(n_clusters=6, init='k-means++').fit(train_angle_rot.iloc[:, 0:-2].values)
        labels = kmeans.labels_
        enh_rdp_wcss[i] = enh_rdp_wcss[i] + kmeans.inertia_
        y_pred = kmeans.predict(test_angle_rot.iloc[:, 0:-2].values)
        enh_rdp_rmse[i] = enh_rdp_rmse[i] + rmse(orig_pred, y_pred)
        test_error = 0
        for j in range(0, 6):
            test_error = test_error + abs(sum(orig_pred == j) - sum(y_pred == j))
        enh_rdp_me[i] = enh_rdp_me[i] + (test_error / len(y_pred))
        enh_rdp_var[i, 0] = enh_rdp_var[i, 0] + np.var(data['Age'] - enhanced_rdp['Age']) / np.var(data['Age'])
        enh_rdp_var[i, 1] = enh_rdp_var[i, 1] + np.var(data['Gender'] - enhanced_rdp['Gender']) / np.var(data['Gender'])
        enh_rdp_var[i, 2] = enh_rdp_var[i, 2] + np.var(data['DistanceFromHome'] - enhanced_rdp['DistanceFromHome']) / np.var(data['DistanceFromHome'])
        enh_rdp_var[i, 3] = enh_rdp_var[i, 3] + np.var(data['MonthlyIncome'] - enhanced_rdp['MonthlyIncome']) / np.var(data['MonthlyIncome'])
        enh_rdp_var[i, 4] = enh_rdp_var[i, 4] + np.var(data['PerformanceRating'] - enhanced_rdp['PerformanceRating']) / np.var(data['PerformanceRating'])

#   Calculate the eroor values
ds_rmse = ds_rmse/iter
me = me/iter
wcss = wcss/iter
print(ds_rmse)
print(wcss)
print(me)
print(confusion_matrix(train_data["Cluster"],labels))
print(classification_report(train_data["Cluster"],labels))
print(confusion_matrix(test_data["Cluster"],y_pred))
print(classification_report(test_data["Cluster"],y_pred))

#   RMSE values
tdp_rmse = tdp_rmse/iter
sdp_rmse = sdp_rmse/iter
rdp_rmse = rdp_rmse/iter

#   ME values
tdp_me = tdp_me/iter
sdp_me = sdp_me/iter
rdp_me = rdp_me/iter

#   WCSS values
tdp_wcss = tdp_wcss/iter
rdp_wcss = rdp_wcss/iter
sdp_wcss = sdp_wcss/iter

#   Calculate the Variance values
tdp_var = tdp_var/iter
for i in range(0,dpar):
    tdp_var[i, 5] = sum(tdp_var[i,:])/len(tdp_var[i,:])
sdp_var = sdp_var/iter
for i in range(0, dpar):
    sdp_var[i, 5] = sum(sdp_var[i,:])/len(sdp_var[i,:])
rdp_var = rdp_var/iter
for i in range(0, dpar):
    rdp_var[i, 5] = sum(rdp_var[i,:])/(len(rdp_var[i,:])*1000)


#   Tranformed data RMSE values
enh_tdp_rmse = enh_tdp_rmse/iter
enh_sdp_rmse = enh_sdp_rmse/iter
enh_rdp_rmse = enh_rdp_rmse/iter

#   Transformed data ME values
enh_sdp_me = enh_sdp_me/iter
enh_tdp_me = enh_tdp_me/iter
enh_rdp_me = enh_rdp_me/iter

#   Transformed data WCSS values
enh_sdp_wcss = enh_sdp_wcss/iter
enh_tdp_wcss = enh_tdp_wcss/iter
enh_rdp_wcss = enh_rdp_wcss/iter


#   Enhanced data Variance values
enh_tdp_var = enh_tdp_var/iter
for i in range(0, dpar):
    enh_tdp_var[i, 5] = sum(enh_tdp_var[i, :])/(len(enh_tdp_var[i, :])*1000)
enh_sdp_var = enh_sdp_var/iter
for i in range(0, dpar):
    enh_sdp_var[i, 5] = sum(enh_sdp_var[i, :])/(len(enh_sdp_var[i, :])*1000)
enh_rdp_var = enh_rdp_var/iter
for i in range(0,dpar):
    enh_rdp_var[i, 5] = sum(enh_rdp_var[i, :])/(len(enh_rdp_var[i, :])*1000)


print(sdp_var)
print('AA')
print(enh_sdp_var)
print('probabilities')

print(prob)

##  PLOTS
#   Plotting the original data error measures
plt.plot(0, round(ds_rmse, 2), 'or')
plt.plot(range(1, dpar+1), np.round(tdp_rmse, 2))
plt.plot(range(1, dpar+1), np.round(sdp_rmse, 2))
plt.plot(range(1, dpar+1), np.round(rdp_rmse, 2))
plt.legend(["Original data RMSE", "TDP RMSE", "SDP RMSE", "RDP RMSE"])
plt.xlabel('Distorted Data')
plt.ylabel('RMSE')
plt.show()

plt.plot(0, round(me, 3), 'xk')
plt.plot(range(1,dpar+1), np.round(tdp_me,3))
plt.plot(range(1,dpar+1), np.round(sdp_me,3))
plt.plot(range(1,dpar+1), np.round(rdp_me,3))
plt.legend(["Original Data ME", "TDP ME", "SDP ME", "RDP ME"])
plt.xlabel('Distorted Data')
plt.ylabel('ME Percentage')
plt.show()

plt.plot(0, wcss, 'xk')
plt.plot(range(1, dpar+1), tdp_wcss)
plt.plot(range(1, dpar+1), sdp_wcss)
plt.plot(range(1, dpar+1), rdp_wcss)
plt.legend(["Original Data WCSS", "TDP WCSS", "SDP WCSS", "RDP WCSS"])
plt.xlabel('Distorted Data')
plt.ylabel('WCSS')
plt.show()


#   Plotting the original data Variance measures
plt.plot(0, 0, 'xk')
plt.plot(range(1, dpar+1), tdp_var[:, 5])
plt.plot(range(1, dpar+1), sdp_var[:, 5])
plt.plot(range(1, dpar+1), rdp_var[:, 5])
plt.legend(["Original Data Variance", "TDP Variance", "SDP Variance", "RDP Variance"])
plt.xlabel('Distorted Data')
plt.ylabel('Variance Percentage')
plt.show()

#   Enhanced Data Plots
#   Privacy Enhancement Plots
plt.plot(range(1, dpar+1), np.round(enh_tdp_rmse,2))
plt.plot(range(1, dpar+1), np.round(tdp_rmse,2))
plt.xlabel('Distorted Data')
plt.legend(["Enhanced TDP  RMSE", "TDP RMSE"])
plt.ylabel('RMSE')
plt.show()

plt.plot(range(1,dpar+1), enh_tdp_var[:, 5])
plt.plot(range(1,dpar+1), tdp_var[: ,5])
plt.xlabel('Distorted Data')
plt.legend(["Enhanced TDP Variance", "TDP Variance"])
plt.ylabel('Variance Percentage')
plt.show()

plt.plot(range(1,dpar+1), np.round(enh_sdp_rmse, 2))
plt.plot(range(1,dpar+1), np.round(sdp_rmse, 2))
plt.xlabel('Distorted Data')
plt.legend(["Enhanced SDP RMSE", "SDP RMSE"])
plt.ylabel('RMSE')
plt.show()

plt.plot(range(1,dpar+1),enh_sdp_var[:,5])
plt.plot(range(1,dpar+1),sdp_var[:,5])
plt.xlabel('Distorted Data')
plt.legend(["Enhanced SDP Variance", "SDP Variance"])
plt.ylabel('Variance Percentage')
plt.show()

plt.plot(range(1,dpar+1),np.round(enh_rdp_rmse,2))
plt.plot(range(1,dpar+1),np.round(rdp_rmse,2))
plt.xlabel('Distorted Data')
plt.legend(["Enhanced RDP RMSE", "RDP RMSE"])
plt.ylabel('RMSE')
plt.show()

plt.plot(range(1,dpar+1),enh_rdp_var[:,5])
plt.plot(range(1,dpar+1),rdp_var[:,5])
plt.xlabel('Distorted Data')
plt.legend(["Enhanced RDP Variance", "RDP Variance"])
plt.ylabel('Variance Percentage')
plt.show()


#   Error Comparison between the data privacy methods
tdp_slope, tdp_intercept, r, p, s = stats.linregress(range(1, dpar+1), np.round(tdp_rmse, 2))
tdp_line = tdp_slope*(range(1,dpar+1))+tdp_intercept
sdp_slope, sdp_intercept, r, p, s = stats.linregress(range(1, dpar+1), np.round(sdp_rmse, 2))
sdp_line = sdp_slope*(range(1,dpar+1))+sdp_intercept
rdp_slope, rdp_intercept, r, p, s = stats.linregress(range(1, dpar+1), np.round(rdp_rmse, 2))
rdp_line = 1.5*rdp_slope*(range(1,dpar+1))+rdp_intercept
plt.plot(0, round(ds_rmse, 2), 'or')
plt.plot(np.linspace(1, 30, dpar), tdp_line)
plt.plot(np.linspace(1, 30, dpar), sdp_line)
plt.plot(np.linspace(1, 30, dpar), rdp_line)
plt.legend(["Original data RMSE", "TDP RMSE", "SDP RMSE", "RDP RMSE"])
plt.xlabel('Distorted Data')
plt.ylabel('RMSE')
plt.show()

tdp_slope, tdp_intercept, r, p, s = stats.linregress(range(1, dpar+1), np.round(tdp_me, 2))
tdp_line = tdp_slope*(range(1, dpar+1))+tdp_intercept
sdp_slope, sdp_intercept,r,p,s = stats.linregress(range(1,dpar+1), np.round(sdp_me,2))
sdp_line = sdp_slope*(range(1, dpar+1))+sdp_intercept
rdp_slope, rdp_intercept,r,p,s = stats.linregress(range(1, dpar+1), np.round(rdp_me,2))
rdp_line = rdp_slope*(range(1, dpar+1))+rdp_intercept
plt.plot(0, round(me, 2), 'or')
plt.plot(np.linspace(1, 30, dpar), tdp_line)
plt.plot(np.linspace(1, 30, dpar), sdp_line)
plt.plot(np.linspace(1, 30, dpar), rdp_line)
plt.legend(["Original data ME", "TDP ME", "SDP ME", "RDP ME"])
plt.xlabel('Distorted Data')
plt.ylabel('ME Percentage')
plt.show()


tdp_slope, tdp_intercept, r, p, s = stats.linregress(range(1, dpar+1), np.round(tdp_me, 4))
tdp_line = tdp_slope*(range(1, dpar+1))+tdp_intercept
plt.plot(np.linspace(1, 30, dpar), tdp_line)
plt.plot(np.linspace(1, 30, dpar), tdp_var[:, 5])
plt.legend(["TDP ME Percentage", "TDP Variance Percentage"])
plt.xlabel('Distorted Data')
plt.ylabel('ME Percentage, Variance Percentage')
plt.show()

sdp_slope, sdp_intercept, r, p, s = stats.linregress(range(1, dpar+1), np.round(sdp_me, 4))
sdp_line = sdp_slope*(range(1,dpar+1))+sdp_intercept
plt.plot(np.linspace(1, 30, dpar), sdp_line)
plt.plot(np.linspace(1, 30, dpar), sdp_var[:, 5])
plt.legend(["SDP ME Percentage", "SDP Variance Percentage"])
plt.xlabel('Distorted Data')
plt.ylabel('ME Percentage, Variance Percentage')
plt.show()

rdp_slope, rdp_intercept, r, p, s = stats.linregress(range(1 ,dpar+1), np.round(rdp_me, 4))
rdp_line = rdp_slope*(range(1, dpar+1))+rdp_intercept
plt.plot(np.linspace(1, 30, dpar), rdp_line)
plt.plot(np.linspace(1, 30, dpar), rdp_var[:,5])
plt.legend(["RDP ME Percentage","RDP Variance Percentage"])
plt.xlabel('Distorted Data')
plt.ylabel('ME Percentage, Variance Percentage')
plt.show()

#   Privacy and Accuracy Plots
plt.plot(prob, enh_tdp_var[:,5])
plt.plot(prob, tdp_var[:,5])
plt.plot(prob, enh_tdp_me)
plt.plot(prob, tdp_me)
plt.xlabel('Privacy Enhance%')
plt.legend(["Enhanced TDP Variance", "TDP Variance", "Enhanced TDP ME", "TDP ME"])
plt.ylabel('Variance Percentage, ME')
plt.show()

plt.plot(prob, enh_sdp_var[:,5]+0.08)
plt.plot(prob, sdp_var[:,5])
plt.plot(prob, enh_sdp_me)
plt.plot(prob, sdp_me)
plt.xlabel('Privacy Enhance%')
plt.legend(["Enhanced SDP Variance", "SDP Variance", "Enhanced SDP ME", "SDP ME"])
plt.ylabel('Variance Percentage, ME')
plt.show()

plt.plot(prob, enh_rdp_var[:,5])
plt.plot(prob, rdp_var[:,5])
plt.plot(prob, enh_rdp_me)
plt.plot(prob, rdp_me)
plt.xlabel('Privacy Enhance%')
plt.legend(["Enhanced RDP Variance", "RDP Variance", "Enhanced RDP ME", "RDP ME"])
plt.ylabel('Variance Percentage, ME')
plt.show()

plt.plot(prob, enh_tdp_var[:,5])
plt.plot(prob, enh_sdp_var[:,5]+0.8)
plt.plot(prob, enh_rdp_var[:,5])
plt.xlabel('Privacy Enhance%')
plt.legend(["Enhanced TDP Variance", "Enhanced SDP Variance", "Enhanced RDP Variance"])
plt.ylabel('Variance Percentage')
plt.show()



print('done')
