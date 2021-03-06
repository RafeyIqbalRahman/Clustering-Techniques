import matplotlib.pyplot as plt
from matplotlib import patches
plt.style.use('ggplot')
import plotly.express as px
from numpy import array
from sklearn.cluster import KMeans

# Create data
x=[11.10, 11.15, 5,
   8, 1, 9,]

y=[2, 8, 1.8,
   8, 0.6, 11,] 

fig = px.scatter(x,
                 y,)
fig.show()

X=array([[1038,660],
         [1045,680],
         [1038,750],
         [897,750],
         [807,780],
         [805,850],])
         
# Checking the optimal number of clusters
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i,
                    init='k-means++', 
                    max_iter=600, 
                    n_init=10, 
                    random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans=KMeans(n_clusters = 2,
              max_iter = 600,
              algorithm = 'auto',
              random_state = 42,)
kmeans.fit(X)

# Finding the center
centroids=kmeans.cluster_centers_

# See the division of data in clusters 0 and 1
labels=kmeans.labels_
print(centroids)
print(labels)
colors=("g.",
        "r.",)
        
for i in range(len(X)):
  print("coordinates:",
        X[i],
        "label:",
        labels[i],)
        
  plt.plot(X[i][0],
           X[i][1],
           colors[labels[i]],
           markersize=10,)
