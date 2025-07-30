# FirstRepoVVM
First github repo off my latest machine learning project

#Step 1: Load the given Banknote authentication dataset.
#done
#Step 2: Calculate statistical measures: mean and standard deviation
#Step 3: Visualise your data as you consider fit.
#Step 4: Evaluate if the dataset is suitable for the K-Means clustering task.
#Step 5: Write a short description of the dataset and your evaluation.

#Meaning of the data:
#V1: Variance of Wavelet Transformed image
#V2: Skewness of Wavelet Transformed image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('datasetProject.csv')
inertias = []

first_Column= data['V1']
second_Column = data['V2']
print("\nTesting output: ")
print(first_Column.head())
print("\n")
print(second_Column.head())

print("\nDisplaying statistics about the dataset...")
v1_mean = np.mean(first_Column)
v2_mean = np.mean(second_Column)
v1_std = np.std(first_Column)
v2_std = np.std(second_Column)

print("\nFirst column: ")
print("Mean, Standard deviation: ", v1_mean, v1_std)
print("\nSecond column: ")
print("Mean, Standard deviation: ", v2_mean, v2_std)

#Trying to test KMeans Clustering on our dataset.
data_matrix = np.column_stack((first_Column, second_Column)) #combining the two datasets into a matrice for computation


#Using the statistical correlation indices
cor_xy = np.corrcoef(first_Column, second_Column)[0,1]
plt.figure(figsize = (16,16))
colors = ['lime', 'purple', 'red']
for k in range(3):
    kmeans = KMeans(n_clusters=k+1).fit(data_matrix)
    labels = kmeans.labels_
    clusters = kmeans.cluster_centers_

    plt.subplot(4, 1, k+1)

   for i in range(k+1):
        cluster_points = data_matrix[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], alpha=0.75, label = f"Cluster {i+1}")



    plt.scatter(clusters[:, 0], clusters[:, 1], c='blue', s=450, marker='*')
    plt.title(f"{k + 1} K-Means run")
    plt.xlabel("Variance of Wavelet Transformed image")
    plt.ylabel("Skewness of Wavelet Transformed image")
    plt.text(0, -14, f"Pearson Correlation Coefficient = {cor_xy:.2f}", fontsize=12)
    plt.legend(loc = "lower right")
    plt.grid(True, alpha=0.6)


#Evaluating the utility for k
for k in range(1, 5):
    model = KMeans(n_clusters=k).fit(data_matrix)
    inertias.append(model.inertia_) #This line calculates the distance score for every point

diff = np.diff(inertias) #Differences between to consecutive distances
diff2 = np.diff(diff) #Second difference
optimal_k = np.argmin(diff2)+2 #Indexing with 2 because we made two differences and lost the indices
#The line above indicates the minimum of distances in that array that has k elements and it will effectively choose the best k for the algorithm

print("\nOptimal k value using Elbow Method for K-Means: ", optimal_k)

plt.subplot(4,1,4)

k_values = range(1,5)
plt.scatter(k_values, inertias, marker = 'o', c='red', s=450)
plt.axvline(x=optimal_k, linestyle='-', c='blue', label = 'Optimal k')
plt.text(3, 17000, f"Optimal value for K = {optimal_k}", fontsize=8)
plt.title("Elbow Method for K-Means", fontsize=13, fontweight='bold')
plt.xlabel("Number of clusters", fontsize=10)
plt.ylabel("Inertia", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




