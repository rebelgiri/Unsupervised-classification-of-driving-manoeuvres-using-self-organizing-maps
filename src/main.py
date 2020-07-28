from get_data_set import *
from training_som_model import *
from silhouette_analysis import *
from label_validation_data_set import *
from plotting import *
import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.cluster import KMeans

path_args = None


def main():
    print('Inside main')
    model_path = Path(path_args.model_export_path)  # Directory path for saving models
    result_path = Path(path_args.work_dir)  # directory path for saving results


    # Please define this function
    data_set = get_data_set()
    data_set = data_set.astype(dtype=float, copy=False)
    print('Data set received...')

   
    # Normalisation of data_set
    data_set_normalized = data_set_normalization(data_set)

    # Split data. The validation data set will be having 80000 samples. Change the value if you want more.
    data_set_to_validate, data_set_to_train = np.array_split(data_set, [80000], axis=0)
    data_set_normalized_to_validate, data_set_normalized_to_train = np.array_split(data_set_normalized, [80000], axis=0)

    print('Shape of training data = ', data_set_normalized_to_train.shape)
    print('Shape of validation data = ', data_set_normalized_to_validate.shape)

    som_shape = (20, 20)
    number_of_features = data_set_normalized_to_train.shape[1]
    sigma_value = 9.0
    learning_rate = 0.5
    neighborhood_function = 'gaussian'
    no_of_iteration = 130000  # 500 times network units
    data_set_name = 'Data set name'

    # Start Training
    som_model = som_training(data_set_normalized_to_train, som_shape, number_of_features, sigma_value, learning_rate,
                             neighborhood_function, no_of_iteration)

    # Save results
    result_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # Plot som map and distance map
    plot_som_map(som_model, data_set_normalized_to_train, data_set_name)
    plot_distance_map(som_model, data_set_name)

    # get the code book or weights of SOM and reshape it
    code_book_of_som = som_model.get_weights()
    code_book_of_som_reshaped = code_book_of_som.reshape(som_shape[0] * som_shape[1], number_of_features)

    no_of_clusters = 3
    # Apply k-means over SOM code book vectors
    som_k_means_trained_model = KMeans(n_clusters=no_of_clusters, random_state=10)
    labels_of_training_data_set = som_k_means_trained_model.fit_predict(code_book_of_som_reshaped)

    # plotting dendrogram
    plot_dendrogram(code_book_of_som_reshaped, data_set_name)

    # Validation data set prediction / Validation
    # first we will get the winner node or BMU for each SOM sample, then we predict the winner node code book
    # vector over k-means model, to determine cluster of a sample

    winner_nodes_of_validation_data_set = np.array(
        [code_book_of_som[som_model.winner(x)] for x in data_set_normalized_to_validate])

    # get the centroid and labels of validation data
    predicted_labels_of_validation_data_set = som_k_means_trained_model.predict(winner_nodes_of_validation_data_set)
    centroids_of_validation_data_set = som_k_means_trained_model.cluster_centers_

    # silhouette analysis over validation data
    silhouette_analysis(winner_nodes_of_validation_data_set,
                        predicted_labels_of_validation_data_set, centroids_of_validation_data_set, no_of_clusters, True)

    # plot training data hexagonal map
    plot_hexagonal_topology(som_model, code_book_of_som_reshaped, labels_of_training_data_set, no_of_clusters,
                            False)

    # plot validation data set hexagonal map
    plot_hexagonal_topology(som_model, winner_nodes_of_validation_data_set, predicted_labels_of_validation_data_set,
                            no_of_clusters,
                            True)

    # plot predicted clusters over validation data using trained model
    plot_predicted_clusters_over_validation_data(data_set_to_validate,
                                                 predicted_labels_of_validation_data_set,
                                                 no_of_clusters)

    # plot predicted clusters over each feature present in the validation data using trained model
    plot_predicted_clusters_over_each_feature_in_validation_data(data_set_to_validate,
                                                                 predicted_labels_of_validation_data_set,
                                                                 no_of_clusters)

    # label validation data
    data_set_to_validate_rounded_two_decimal_points = np.round(data_set_to_validate, 2)
    data_set_to_validate_string_labels = np.array(
        [label_validation_data_by_strings(x) for x in data_set_to_validate_rounded_two_decimal_points])

    data_set_to_validate_number_labels = np.array(
        [label_validation_data_by_numbers(x) for x in data_set_to_validate_rounded_two_decimal_points])

    data_set_to_validate_number_labels_reshaped = np.reshape(data_set_to_validate_number_labels, (80000, 1))
    data_set_to_validate_string_labels_reshaped = np.reshape(data_set_to_validate_string_labels, (80000, 1))

    validation_data_set_with_labels = np.concatenate((data_set_to_validate_rounded_two_decimal_points,
                                                      data_set_to_validate_string_labels_reshaped,
                                                      data_set_to_validate_number_labels_reshaped), axis=1)
    np.savetxt('validation_data_set_with_labels.csv',
               validation_data_set_with_labels,
               fmt="%s",
               delimiter=",")

    print('Shape of validation data set with labels = ', validation_data_set_with_labels.shape)
    som_logs = open('Som_logs.txt', 'a')
    print('Shape of validation data set with labels = ', validation_data_set_with_labels.shape, file=som_logs)
    Cluster0 = np.array([data_set_to_validate_number_labels_reshaped[predicted_labels_of_validation_data_set == 0]])
    Cluster1 = np.array([data_set_to_validate_number_labels_reshaped[predicted_labels_of_validation_data_set == 1]])
    Cluster2 = np.array([data_set_to_validate_number_labels_reshaped[predicted_labels_of_validation_data_set == 2]])

    Stop_Driving = np.array(
        data_set_to_validate_number_labels_reshaped[data_set_to_validate_number_labels_reshaped == 0])
    Start_Driving = np.array(
        data_set_to_validate_number_labels_reshaped[data_set_to_validate_number_labels_reshaped == 1])
    Accelerate_in_the_lane = np.array(
        data_set_to_validate_number_labels_reshaped[data_set_to_validate_number_labels_reshaped == 2])
    Decelerate_in_the_lane = np.array(
        data_set_to_validate_number_labels_reshaped[data_set_to_validate_number_labels_reshaped == 3])
    Drive_with_constant_speed = np.array(
        data_set_to_validate_number_labels_reshaped[data_set_to_validate_number_labels_reshaped == 4])
    Lane_change_to_the_left = np.array(
        data_set_to_validate_number_labels_reshaped[data_set_to_validate_number_labels_reshaped == 5])
    Lane_change_to_the_right = np.array(
        data_set_to_validate_number_labels_reshaped[data_set_to_validate_number_labels_reshaped == 6])

    print('No. of Stop Driving Manoeuvres in validation dataset = ', Stop_Driving.shape[0], file=som_logs)
    print('No. of Start Driving Manoeuvres in validation dataset = ', Start_Driving.shape[0], file=som_logs)
    print('No. of Accelerate in the lane Manoeuvres in validation dataset = ', Accelerate_in_the_lane.shape[0],
          file=som_logs)
    print('No. of Decelerate in the lane Manoeuvres in validation dataset = ', Decelerate_in_the_lane.shape[0],
          file=som_logs)
    print('No. of Drive with constant speed Manoeuvres in validation dataset = ', Drive_with_constant_speed.shape[0],
          file=som_logs)
    print('No. of Lane change to the left Manoeuvres in validation dataset = ', Lane_change_to_the_left.shape[0],
          file=som_logs)
    print('No. of Lane change to the right Manoeuvres in validation dataset = ', Lane_change_to_the_right.shape[0],
          file=som_logs)

    print('Samples in Cluster 0 ', file=som_logs)
    print(Cluster0.shape[1], file=som_logs)
    print('Samples in Cluster 1 ', file=som_logs)
    print(Cluster1.shape[1], file=som_logs)
    print('Samples in Cluster 2 ', file=som_logs)
    print(Cluster2.shape[1], file=som_logs)

    print('######################################Cluster0######################################', file=som_logs)
    unique, counts = np.unique(Cluster0, return_counts=True)
    print(dict(zip(unique, counts)), file=som_logs)
    for unique, counts in zip(unique, counts):
        if unique == 0:
            print('Stop Driving = %d of %d samples in Cluster 0' % (counts, Cluster0.shape[1]), file=som_logs)
            print('Stop Driving in percentage = %.2f  in the cluster' % ((counts / Cluster0.shape[1]) * 100),
                  file=som_logs)
            print('Stop Driving in percentage = %.2f  of total Stop Driving manoeuvres' %
                  ((counts / Stop_Driving.shape[0]) * 100), file=som_logs)
        elif unique == 1:
            print('Start Driving = %d of %d samples in Cluster 0' % (counts, Cluster0.shape[1]), file=som_logs)
            print('Start Driving in percentage = %.2f  in the cluster' % ((counts / Cluster0.shape[1]) * 100),
                  file=som_logs)
            print('Start Driving in percentage = %.2f  of total Start Driving manoeuvres' %
                  ((counts / Start_Driving.shape[0]) * 100), file=som_logs)
        elif unique == 2:
            print('Accelerate in the lane = %d of %d samples in Cluster 0' % (counts, Cluster0.shape[1]), file=som_logs)
            print('Accelerate in the lane in percentage = %.2f  in the cluster' % ((counts / Cluster0.shape[1]) * 100),
                  file=som_logs)
            print('Accelerate in the lane in percentage = %.2f  of total Accelerate in the lane manoeuvres' %
                  ((counts / Accelerate_in_the_lane.shape[0]) * 100), file=som_logs)
        elif unique == 3:
            print('Decelerate in the lane = %d of %d samples in Cluster 0' % (counts, Cluster0.shape[1]), file=som_logs)
            print('Decelerate in the lane in percentage = %.2f  in the cluster' % ((counts / Cluster0.shape[1]) * 100),
                  file=som_logs)
            print('Decelerate in the lane in percentage = %.2f  of total Decelerate in the lane manoeuvres' %
                  ((counts / Decelerate_in_the_lane.shape[0]) * 100), file=som_logs)
        elif unique == 4:
            print('Drive with constant speed = %d of %d samples in Cluster 0' % (counts, Cluster0.shape[1]),
                  file=som_logs)
            print(
                'Drive with constant speed in percentage = %.2f in the cluster' % ((counts / Cluster0.shape[1]) * 100),
                file=som_logs)
            print('Drive with constant speed in percentage = %.2f  of total Drive with constant speed manoeuvres' %
                  ((counts / Drive_with_constant_speed.shape[0]) * 100), file=som_logs)
        elif unique == 5:
            print('Lane change to the left = %d of %d samples in Cluster 0 ' % (counts, Cluster0.shape[1]),
                  file=som_logs)
            print('Lane change to the left in percentage = %.2f in the cluster' % ((counts / Cluster0.shape[1]) * 100),
                  file=som_logs)
            print('Lane change to the left in percentage = %.2f of total Lane change to the left manoeuvres' %
                  ((counts / Lane_change_to_the_left.shape[0]) * 100), file=som_logs)
        else:
            print('Lane change to the right = %d of %d samples in Cluster 0' % (counts, Cluster0.shape[1]),
                  file=som_logs)
            print(
                'Lane change to the right in percentage = %.2f  in the cluster' % ((counts / Cluster0.shape[1]) * 100),
                file=som_logs)
            print('Lane change to the right in percentage = %.2f  of total Lane change to the right manoeuvres' %
                  ((counts / Lane_change_to_the_right.shape[0]) * 100), file=som_logs)

    print('######################################Cluster1######################################', file=som_logs)
    unique, counts = np.unique(Cluster1, return_counts=True)
    print(dict(zip(unique, counts)), file=som_logs)
    for unique, counts in zip(unique, counts):
        if unique == 0:
            print('Stop Driving = %d of %d samples in Cluster 1' % (counts, Cluster1.shape[1]), file=som_logs)
            print('Stop Driving in percentage = %.2f  in the cluster' % ((counts / Cluster1.shape[1]) * 100),
                  file=som_logs)
            print('Stop Driving in percentage = %.2f  of total Stop Driving manoeuvres' %
                  ((counts / Stop_Driving.shape[0]) * 100), file=som_logs)
        elif unique == 1:
            print('Start Driving = %d of %d samples in Cluster 1' % (counts, Cluster1.shape[1]), file=som_logs)
            print('Start Driving in percentage = %.2f  in the cluster' % ((counts / Cluster1.shape[1]) * 100),
                  file=som_logs)
            print('Start Driving in percentage = %.2f  of total Start Driving manoeuvres' %
                  ((counts / Start_Driving.shape[0]) * 100), file=som_logs)
        elif unique == 2:
            print('Accelerate in the lane = %d of %d samples in Cluster 1' % (counts, Cluster1.shape[1]), file=som_logs)
            print('Accelerate in the lane in percentage = %.2f  in the cluster' % ((counts / Cluster1.shape[1]) * 100),
                  file=som_logs)
            print('Accelerate in the lane in percentage = %.2f  of total Accelerate in the lane manoeuvres' %
                  ((counts / Accelerate_in_the_lane.shape[0]) * 100), file=som_logs)
        elif unique == 3:
            print('Decelerate in the lane = %d of %d samples in Cluster 1' % (counts, Cluster1.shape[1]), file=som_logs)
            print('Decelerate in the lane in percentage = %.2f  in the cluster' % ((counts / Cluster1.shape[1]) * 100),
                  file=som_logs)
            print('Decelerate in the lane in percentage = %.2f  of total Decelerate in the lane manoeuvres' %
                  ((counts / Decelerate_in_the_lane.shape[0]) * 100), file=som_logs)
        elif unique == 4:
            print('Drive with constant speed = %d of %d samples in Cluster 1' % (counts, Cluster1.shape[1]),
                  file=som_logs)
            print(
                'Drive with constant speed in percentage = %.2f  in the cluster' % ((counts / Cluster1.shape[1]) * 100),
                file=som_logs)
            print('Drive with constant speed in percentage = %.2f  of total Drive with constant speed manoeuvres' %
                  ((counts / Drive_with_constant_speed.shape[0]) * 100), file=som_logs)
        elif unique == 5:
            print('Lane change to the left = %d of %d samples in Cluster 1' % (counts, Cluster1.shape[1]),
                  file=som_logs)
            print('Lane change to the left in percentage = %.2f  in the cluster' % ((counts / Cluster1.shape[1]) * 100),
                  file=som_logs)
            print('Lane change to the left in percentage = %.2f  of total Lane change to the left manoeuvres' %
                  ((counts / Lane_change_to_the_left.shape[0]) * 100), file=som_logs)
        else:
            print('Lane change to the right = %d of %d samples in Cluster 1' % (counts, Cluster1.shape[1]),
                  file=som_logs)
            print(
                'Lane change to the right in percentage = %.2f  in the cluster' % ((counts / Cluster1.shape[1]) * 100),
                file=som_logs)
            print('Lane change to the right in percentage = %.2f  of total Lane change to the right manoeuvres' %
                  ((counts / Lane_change_to_the_right.shape[0]) * 100), file=som_logs)

    print('######################################Cluster2######################################', file=som_logs)
    unique, counts = np.unique(Cluster2, return_counts=True)
    print(dict(zip(unique, counts)), file=som_logs)
    for unique, counts in zip(unique, counts):
        if unique == 0:
            print('Stop Driving = %d of %d samples in Cluster 2' % (counts, Cluster2.shape[1]), file=som_logs)
            print('Stop Driving in percentage = %.2f  in the cluster' % ((counts / Cluster2.shape[1]) * 100),
                  file=som_logs)
            print('Stop Driving in percentage = %.2f  of total Stop Driving manoeuvres' %
                  ((counts / Stop_Driving.shape[0]) * 100), file=som_logs)
        elif unique == 1:
            print('Start Driving = %d of %d samples in Cluster 2' % (counts, Cluster2.shape[1]), file=som_logs)
            print('Start Driving in percentage = %.2f  in the cluster' % ((counts / Cluster2.shape[1]) * 100),
                  file=som_logs)
            print('Start Driving in percentage = %.2f  of total Start Driving manoeuvres' %
                  ((counts / Start_Driving.shape[0]) * 100), file=som_logs)
        elif unique == 2:
            print('Accelerate in the lane = %d of %d samples in Cluster 2' % (counts, Cluster2.shape[1]), file=som_logs)
            print('Accelerate in the lane in percentage = %.2f  in the cluster' % ((counts / Cluster2.shape[1]) * 100),
                  file=som_logs)
            print('Accelerate in the lane in percentage = %.2f  of total Accelerate in the lane manoeuvres' %
                  ((counts / Accelerate_in_the_lane.shape[0]) * 100), file=som_logs)
        elif unique == 3:
            print('Decelerate in the lane = %d of %d samples in Cluster 2' % (counts, Cluster2.shape[1]), file=som_logs)
            print('Decelerate in the lane in percentage = %.2f  in the cluster' % ((counts / Cluster2.shape[1]) * 100),
                  file=som_logs)
            print('Decelerate in the lane in percentage = %.2f  of total Decelerate in the lane manoeuvres' %
                  ((counts / Decelerate_in_the_lane.shape[0]) * 100), file=som_logs)
        elif unique == 4:
            print('Drive with constant speed = %d of %d samples in Cluster 2' % (counts, Cluster2.shape[1]),
                  file=som_logs)
            print(
                'Drive with constant speed in percentage = %.2f in the cluster' % ((counts / Cluster2.shape[1]) * 100),
                file=som_logs)
            print('Drive with constant speed in percentage = %.2f of total Drive with constant speed manoeuvres' %
                  ((counts / Drive_with_constant_speed.shape[0]) * 100), file=som_logs)
        elif unique == 5:
            print('Lane change to the left = %d of %d samples in Cluster 2' % (counts, Cluster2.shape[1]),
                  file=som_logs)
            print('Lane change to the left in percentage = %.2f  in the cluster' % ((counts / Cluster2.shape[1]) * 100),
                  file=som_logs)
            print('Lane change to the left in percentage = %.2f  of total Lane change to the left manoeuvres' %
                  ((counts / Lane_change_to_the_left.shape[0]) * 100), file=som_logs)
        else:
            print('Lane change to the right = %d of %d samples in Cluster' % (counts, Cluster2.shape[1]), file=som_logs)
            print(
                'Lane change to the right in percentage = %.2f  in the cluster' % ((counts / Cluster2.shape[1]) * 100),
                file=som_logs)
            print('Lane change to the right in percentage = %.2f  of total Lane change to the right manoeuvres' %
                  ((counts / Lane_change_to_the_right.shape[0]) * 100), file=som_logs)

    som_logs.close()

    # comparison between defined manoeuvres and predicted labels
    comparison_between_defined_manoeuvres_and_predicted_labels(som_model, data_set_normalized_to_validate,
                                                               data_set_to_validate_number_labels)

    # Saving the som in the file som_model.p
    with open('som_model.p', 'wb') as outfile:
        pickle.dump(som_model, outfile)

    # model can be loaded as follows
    # with open('som_model.p', 'rb') as infile:
    # som_model = pickle.load(infile)
    # Note that if a lambda function is used to define the decay factor MiniSom will not be pickable anymore.

    # Saving the final model in the file som_k_means_trained_model.p
    with open('som_k_means_trained_model.p', 'wb') as outfile:
        pickle.dump(som_k_means_trained_model, outfile)

    # model can be loaded as follows
    # with open('som_k_means_trained_model.p', 'rb') as infile:
    # som_k_means_trained_model = pickle.load(infile)

    # plotting quantization error and topographic error
    # take too much time for execution
    # plot_quantization_topographic_error(som_model, data_set_normalized_to_train, no_of_iteration)

    no_of_cluster_list = [2, 3, 4, 5, 6, 7, 8, 9]
    silhouette_avg_list = []
    for i in no_of_cluster_list:
        # Apply k-means over SOM code book vectors
        som_k_means_trained_model = KMeans(n_clusters=i, random_state=10)
        trained_data_labels = som_k_means_trained_model.fit_predict(code_book_of_som_reshaped)
        trained_data_labels_centroids = som_k_means_trained_model.cluster_centers_
        # silhouette analysis
        silhouette_avg_list.append(
            silhouette_analysis(code_book_of_som_reshaped, trained_data_labels, trained_data_labels_centroids,
                                i, False))

    plot_silhouette_avg_vs_clusters(silhouette_avg_list, no_of_cluster_list)

    print('End...')


if __name__ == "__main__":
    print('Start...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_export_path', type=str,
                        default='/model/to-deploy',
                        help='Path where the trained model will be saved.')
    parser.add_argument('--work_dir', type=str,
                        default='/model/work',
                        help='Path to working directory')

    path_args = parser.parse_args()

    exit(main())
