#!/usr/bin/env python
# coding: utf-8

# In[37]:


from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder


# In[38]:


df = pd.read_csv("Mall_customers.csv")
df.head()


# In[5]:


print(df.columns)


# In[39]:


X = df[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']]
label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])


# In[23]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], alpha=0.5)
plt.title('Annual Income vs. Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.grid(True)
plt.show()


# In[40]:


inertia=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state = 42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
#plot the elbow curve
plt.plot(range(1,11),inertia)
plt.xlabel('no_of_cluster')
plt.ylabel('inertia')
plt.show()


# In[41]:


# From the elbow curve, choose the optimal k=4
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=42)
kmeans.fit(X)
# Create a new DataFrame with 'ID' and 'Cluster' columns
result_df = pd.DataFrame({'CustomerID': df['CustomerID'], 'Cluster': kmeans.labels_})

# Save the new DataFrame to a CSV file
result_df.to_csv('clustered_data.csv', index=False)


# In[42]:


plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=300, c='red', marker='x')
plt.title('K-means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:




