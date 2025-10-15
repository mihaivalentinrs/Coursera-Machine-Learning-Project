
🏦 Banknote Authentication – K-Means Clustering Analysis
📘 Project Description

This project performs an exploratory data analysis and clustering evaluation on the Banknote Authentication dataset. The goal is to understand the statistical characteristics of the data and determine whether the dataset is suitable for K-Means clustering.

The dataset contains features extracted from images of genuine and forged banknotes using Wavelet Transforms. For simplicity, this project focuses on two key attributes:

V1: Variance of Wavelet Transformed image

V2: Skewness of Wavelet Transformed image

⚙️ Steps Implemented
1. Load the Dataset

The script imports the dataset (datasetProject.csv) into a Pandas DataFrame for further analysis.

2. Compute Statistical Measures

Basic statistical indicators are calculated for the two numerical features:

Mean

Standard Deviation

These metrics give an overview of the data’s central tendency and dispersion.

3. Data Visualization

Scatter plots are generated to visualize the relationship between V1 and V2, showing how data points are distributed. Multiple K-Means cluster configurations (k = 1, 2, 3) are plotted to visualize clustering tendencies.

Additionally, the Pearson correlation coefficient is computed to evaluate the linear relationship between the two features.

4. K-Means Clustering & Elbow Method

The K-Means algorithm is applied with varying numbers of clusters (k = 1–4).
The Elbow Method is then used to determine the optimal value of k, based on the inertia scores (sum of squared distances to cluster centers).

5. Evaluation

By examining the Elbow plot and the structure of the clusters, we can evaluate whether the dataset exhibits clear separability — a key requirement for effective K-Means clustering.

📊 Results Summary

Statistical Insights:
The mean and standard deviation values reveal that the data is reasonably spread out, indicating variability that could support clustering.

Correlation:
The Pearson correlation coefficient quantifies how linearly related V1 and V2 are.

Optimal K:
The Elbow Method identifies an optimal number of clusters, suggesting a natural separation pattern in the data.

🧠 Evaluation of Suitability for K-Means

The Banknote Authentication dataset shows distinguishable groupings in its feature space, meaning K-Means is moderately suitable for exploratory analysis.
However, since the original dataset is supervised (it has class labels for “authentic” and “forged”), K-Means can only provide unsupervised insights and should not be used for final classification without comparing against true labels.

📦 Technologies Used

  Python

  NumPy

  Pandas

  Matplotlib

  scikit-learn

📈 Example Output

  Scatter plots showing K-Means clusters for k = 1, 2, and 3

  Elbow plot identifying the optimal number of clusters

  Statistical summary of V1 and V2

  Pearson correlation coefficient displayed on plots

🧾 Author

  Developed for educational purposes to demonstrate data exploration, statistical analysis, and unsupervised learning (K-Means)     using the Banknote Authentication dataset.
