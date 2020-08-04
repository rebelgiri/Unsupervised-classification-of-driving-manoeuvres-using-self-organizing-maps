# Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps

## Abstarct
This work proposes an unsupervised method to classify highway driving manoeuvres using the self-organizing map (SOM). The SOM is an excellent method during the initial phase of data mining, it projects high dimensional input space onto the low dimensional regular map of nodes that can be effectively utilized to visualize, classify, cluster, and explore properties of the data. The low dimensionality of the resulting map allows for a graphical presentation of the data which can be easily interpreted by humans. This project uses multivariate time-series data collected over highways and it has pre-processed using min-max scaling and fed into the SOM algorithm to train an artificial neural network. This work is a collection of series of experiments in which it has been attempted to visualize clusters using distance maps, cluster maps, and the model is validated using the validation data set.

## SOM

![alt text](https://github.com/rebelgiri/Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps/blob/master/results/Clusters.png)
The SOM is a representation of small clusters formed after SOM training. In this work, a SOM of rectangular shape with a 20 x 20 grid of nodes (400 nodes) has been trained.

## Distance map

![alt text](https://github.com/rebelgiri/Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps/blob/master/results/Distance_Map.png)
The distance map is a representation of Euclidean distance between the neighbouring nodes depicted in a two-dimensional image. It helps to visualize the number of clusters in data set. A dark colouring between the nodes corresponds to a large distance and thus a gap between the samples in the input space. A light colouring
between the nodes signifies that the samples are close to each other in the input space. Light areas can be thought of as clusters and dark areas as cluster separators. This can
be a helpful presentation when one tries to find clusters in the input data without having any a priori information about the clusters. The colour scale beside the distance
map describes the distance between nodes.

## SOM clustering
![alt text](https://github.com/rebelgiri/Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps/blob/master/results/Hexagonal%20Map%20Training%20Data.png)
It is a concrete representation of major clusters identified after applying the k-means clustering algorithm over the SOM.

## Hierarchical clustering

![alt text](https://github.com/rebelgiri/Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps/blob/master/results/dendrogram.png)
![alt text](https://github.com/rebelgiri/Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps/blob/master/results/dendrogram_truncated.png)

It is possible to visualize the tree representing the hierarchical merging of clusters as a dendrogram. The dendrogram is a type of tree diagram showing hierarchical clustering. It represents relationships between similar sets of samples. The vertical axis of the dendrogram represents the distance between clusters. The horizontal axis represents the samples and clusters.

## Validation
### Clustering of driving manoeuvres
![alt text](https://github.com/rebelgiri/Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps/blob/master/results/Hexagonal%20Map%20Unseen%20Data.png)


### Classification of driving manoeuvres
![alt text](https://github.com/rebelgiri/Unsupervised-classification-of-driving-manoeuvres-using-self-organizing-maps/blob/master/results/Classification_of_driving_manoeuvres.png)

