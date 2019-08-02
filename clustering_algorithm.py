import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

data = pd.read_csv("WICC Mentorship Responses.csv")
data.fillna('N/A', inplace = True)
data.drop(['Major', 'Minor', 'School', 'What do you like to do for fun?'], 1, inplace = True)
lb_make = LabelEncoder()
data["Academic Interests"] = lb_make.fit_transform(data["Academic Interests in Computing"])
data[["Academic Interests in Computing", "Academic Interests"]]
data["Mentor_or_Mentee"] = lb_make.fit_transform(data["Mentor or Mentee"])
data[["Mentor or Mentee", "Mentor_or_Mentee"]]
data.drop(['Career interests in CS', 'Academic Interests in Computing', 'How often do you want to meet?', 'Mentor or Mentee', 'Year', 'Unnamed: 9'], 1, inplace = True)

kmeans = KMeans(n_clusters=18, random_state=0).fit(data)

# Get the cluster centroids
print(kmeans.cluster_centers_)

# Get the cluster labels
print(kmeans.labels_)

# Plotting the data points on a 2D plane
plt.scatter(data['Academic Interests'], data['Mentor_or_Mentee'], color='r')
plt.title('WICC Mentorship Groups')
plt.show()

# Calculate silhouette_score: 0.571501063915
print(silhouette_score(data, kmeans.labels_))
