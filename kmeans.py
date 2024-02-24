

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial import distance_matrix

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

data1 = pd.read_csv("/content/Wholedata.csv")

data1.dropna(inplace=True)

features = ['Spend [USD]', '# of Impressions', 'Reach',
       '# of Website Clicks', '# of Searches', '# of View Content',
       '# of Add to Cart', '# of Purchase']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data1[features])

data_scaled[1:5]

# Choose the number of clusters (for example, 3 clusters)
k = 2

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, algorithm= "lloyd")
data1['cluster'] = kmeans.fit_predict(data_scaled)


cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_labels = kmeans.labels_

data_class =["Control Campaign","Test Campaign"]
dc =dict(zip(data_class,range(0,2)))
print(dc)

data1["mapped"] = data1["Campaign Name"]

data1["mapped"] = data1["mapped"].map(dc)

data_scaled[1]

dataset = pd.DataFrame({'Spend': data_scaled[:, 0],
                        'No of Impressions': data_scaled[:, 1],
                        'Reach': data_scaled[:, 2],
                        '# of searches': data_scaled[:, 3], '# of Views': data_scaled[:, 4]})

conf_matrix = confusion_matrix(data1["mapped"], data1["cluster"])

# Displaying the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])

import matplotlib.pyplot as plt
cm_display.plot()
plt.show()

Accuracy = metrics.accuracy_score(data1["mapped"], data1["cluster"])
Accuracy*100

silhouette_score(data_scaled, data1["cluster"])

Silhouette_Score =[]
for i in range(2,7):

  kmeans = KMeans(n_clusters=i, algorithm= "lloyd",n_init='auto')
  data1['cluster'] = kmeans.fit_predict(data_scaled)
  Silhouette_Score.append(silhouette_score(data_scaled, data1["cluster"]))
  cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
  print("for I=",i,silhouette_score(data_scaled, data1["cluster"]))

# Commented out IPython magic to ensure Python compatibility.
for n_clusters in range(2,6):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data_scaled) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters,n_init='auto')
    cluster_labels = clusterer.fit_predict(data_scaled)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data_scaled, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        data_scaled[:, 1], data_scaled[:, 3], c=cluster_labels
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
#         % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()

wcss =[]
for i in range(1, 6):
    kmeans = KMeans(n_clusters = i, n_init='auto', random_state = 42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title("Using Elbow Method")
plt.show()

plt.plot([2,3,4,5,6], Silhouette_Score)
plt.title("Using Silhouette Score")

dataset.isnull().sum()

dataset.head(10)

dataset.insert(0, column="Campaign Name",value= data1["Campaign Name"])

dataset.dropna(inplace=True)

dataset.set_index("Campaign Name", inplace=True)

dataset1 = dataset.head(10)
dataset1 = dataset1.append(dataset.tail(10))

df = pd.DataFrame(distance_matrix(dataset.values, dataset.values), index=dataset.index, columns=dataset.index)

import seaborn as sns

df =df.where(np.tril(np.ones(df.shape)).astype(np.bool))

newd = df.iloc[0:5,0:5]
newd = newd.append(df.iloc[54:59,0:5])

plt.figure(figsize=(20, 8))
sns.heatmap(newd, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
plt.title('Clutering Distance Matrix')
plt.show()

