import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.express as px
from numpy import array, unique, where
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering

# Create data
x_arr = array([11.10, 11.15,
               5, 8, 1, 9,])
      
y_arr =  array([2, 8, 1.8,
                8, 0.6, 11,])

# Convert array to a DataFrame
x = DataFrame(x_arr)
y = DataFrame(y_arr) 

fig = px.scatter(x_arr, y_arr,)
fig.show()

X=array([[1038,660],
         [1045,680],
         [1038,750],
         [897,750],
         [807,780],
         [805,850],])
         
# Fitting AgglomerativeClustering model to the dataset
model = AgglomerativeClustering(n_clusters=2)
model.fit(X)

# See the division of data in clusters 0 and 1
y = model.fit_predict(X)
y

clusters = unique(y)

for cluster in clusters:
	row_ix = where(y == cluster)
	plt.scatter(X[row_ix, 0], 
              X[row_ix, 1],)
plt.show()
