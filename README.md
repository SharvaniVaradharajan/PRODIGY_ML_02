# Customer Segmentation using K-means Clustering

This project demonstrates the use of the K-means clustering algorithm to segment customers of a retail store based on their purchase history. The primary goal is to group customers into distinct clusters to better understand their behavior and tailor marketing strategies accordingly.

## Project Overview

We use the K-means clustering algorithm to group customers based on two key features:
- `Annual Income (k$)`: The annual income of the customer in thousands of dollars.
- `Spending Score (1-100)`: A score assigned to the customer based on their spending behavior.

## Dataset

- `Mall_Customers.csv`: The dataset containing features for clustering.
- `kmeans_clustering.py`: Python script implementing the K-means clustering algorithm.
  
## Dependencies

- Python 3.x
- pandas
- matplotlib
- scikit-learn
## Explanation
#### Step 1: Import Libraries
- `pandas`: Used for data manipulation and analysis. Here, it helps in loading and handling the dataset.
- `matplotlib.pyplot`: Used for plotting graphs and visualizing data.
- `sklearn.cluster.KMeans`: The K-means clustering algorithm implementation from scikit-learn.
#### Step 2: Load the Dataset
- `pd.read_csv('Mall_Customers.csv')`: Loads the dataset from a CSV file into a pandas DataFrame.

#### Step 3: Extract Features

- `df[['Annual Income (k$)', 'Spending Score (1-100)']]`: Selects the columns `Annual Income (k$)` and `Spending Score (1-100)` from the DataFrame. These are the features used for clustering.

#### Step 4: Determine the Optimal Number of Clusters Using the Elbow Method

- `wcss = []`: Initializes an empty list to store the within-cluster sum of squares (WCSS) for different numbers of clusters.
- `for i in range(1, 11)`: Iterates over a range of cluster numbers from 1 to 10.
  - `KMeans(n_clusters=i, ...)`: Initializes the K-means algorithm with `i` clusters.
  - `kmeans.fit(X)`: Fits the K-means algorithm to the data `X`.
  - `wcss.append(kmeans.inertia_)`: Appends the WCSS (inertia) of the current clustering to the list `wcss`.
- `plt.plot(range(1, 11), wcss)`: Plots the WCSS against the number of clusters.
- `plt.title('Elbow Method')`, `plt.xlabel('Number of clusters')`, `plt.ylabel('WCSS')`: Sets the title and labels of the plot.
- `plt.show()`: Displays the plot.

The "elbow" point on the plot indicates the optimal number of clusters.

#### Step 5: Apply K-means to the Dataset

- `KMeans(n_clusters=5, ...)`: Initializes the K-means algorithm with 5 clusters (assuming the Elbow Method suggested this as the optimal number).
- `y_kmeans = kmeans.fit_predict(X)`: Fits the K-means algorithm to the data `X` and predicts the cluster indices for each data point. The result, `y_kmeans`, is an array of cluster labels for each customer.

#### Step 6: Visualize the Clusters
- `plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_kmeans, cmap='viridis')`: Creates a scatter plot of the data points, coloring them based on their cluster labels (`y_kmeans`). The `cmap='viridis'` argument specifies the color map to use.
- `plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')`: Plots the cluster centers as red 'x' marks. The `s=300` argument specifies the size of the markers.
- `plt.title('K-means Clustering')`, `plt.xlabel('Annual Income')`, `plt.ylabel('Spending Score')`: Sets the title and labels of the plot.
- `plt.show()`: Displays the plot.
## Results

The results are visualized in a scatter plot, showing the clusters of customers based on their annual income and spending score. Each cluster is color-coded, and the cluster centers are highlighted.

## Acknowledgments

The dataset used in this project is provided by Kaggle.
