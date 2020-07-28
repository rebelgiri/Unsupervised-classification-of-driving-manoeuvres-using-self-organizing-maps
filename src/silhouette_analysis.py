import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np


def silhouette_analysis(data, labels, centroids, no_of_clusters, validation_flag):
    som_logs = open('Som_logs.txt', 'a')

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.20, 10.80))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (no_of_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, labels)

    if validation_flag:
        print("For number of clusters =", no_of_clusters,
              "The average silhouette score for validation data is :", silhouette_avg)
        print("For number of clusters =", no_of_clusters,
              "The average silhouette score for validation data is :", silhouette_avg, file=som_logs)
    else:
        print("For number of clusters =", no_of_clusters,
              "The average silhouette score is :", silhouette_avg)
        print("For number of clusters =", no_of_clusters,
              "The average silhouette score is :", silhouette_avg, file=som_logs)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, labels)

    y_lower = 10
    for i in range(no_of_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / no_of_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", fontdict={'fontsize': 14})
    ax1.set_xlabel("The silhouette coefficient values", fontsize=12)
    ax1.set_ylabel("Cluster label", fontsize=12)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / no_of_clusters)
    ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Draw white circles at cluster centers
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centroids):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.", fontdict={'fontsize': 14})
    ax2.set_xlabel("Feature space for the 1st feature", fontsize=12)
    ax2.set_ylabel("Feature space for the 2nd feature", fontsize=12)

    if validation_flag:
        plt.suptitle(("Silhouette analysis for clustering on validation data "
                      "with number of clusters = %d" % no_of_clusters),
                     fontsize=14, fontweight='bold')

        fig.savefig("Silhouette analysis for clustering on validation data "
                    "with number of clusters = %d" % no_of_clusters, dpi=1200)
    else:
        plt.suptitle(("Silhouette analysis for clustering on data "
                      "with number of clusters = %d" % no_of_clusters),
                     fontsize=14, fontweight='bold')

        fig.savefig("Silhouette analysis for clustering on data "
                    "with number of clusters = %d" % no_of_clusters, dpi=1200)

    som_logs.close()
    return silhouette_avg
