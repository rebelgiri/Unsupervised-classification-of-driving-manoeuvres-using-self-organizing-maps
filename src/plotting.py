import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import sys
from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

feature_names = ['Speed',
                 'Longitudinal acceleration',
                 'Lateral acceleration', 'Steering angle',
                 'Yaw angle', 'Yaw rate',
                 'Brake pedal', 'Accelerator pedal', 'Lateral distance'
                 ]

color_list = ['b',
              'g',
              'r', 'c',
              'm', 'y',
              'tab:brown', 'tab:orange', 'tab:orange', 'tab:purple'
              ]


def plot_som_map(som_model, data_set_to_train, data_set_name):
    # Returns a matrix where the element i,j is the number of times
    # that the neuron i,j have been winner
    frequencies = som_model.activation_response(data_set_to_train)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))

    # Plot map of clusters
    c = ax.pcolor(frequencies.T, edgecolors='k', cmap='RdBu', linewidths=1)
    cb = fig.colorbar(c, ax=ax)
    # ax.set_title(data_set_name + ' Map', fontdict={'fontsize': 25})
    ax.set_title('SOM Map', fontdict={'fontsize': 25})
    cb.set_label(label='No. of samples at each node', fontdict={'fontsize': 20})
    cb.ax.tick_params(labelsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig('Clusters.png', dpi=1200)


def plot_distance_map(som_model, data_set_name):
    # distance map
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
    c = ax.pcolor(som_model.distance_map().T, cmap='Blues')
    # ax.set_title(data_set_name + ' Distance Map', fontdict={'fontsize': 25})
    ax.set_title('Distance Map', fontdict={'fontsize': 25})
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(label='Distance from nodes in the neighbourhood', fontdict={'fontsize': 20})
    cb.ax.tick_params(labelsize=18)
    ax.set_yticks([])
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('Distance_Map.png', dpi=1200)


# dendrogram plotting
def plot_dendrogram(code_book_of_som_reshaped, data_set_name):
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
    linked = linkage(code_book_of_som_reshaped, 'centroid')

    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               ax=ax1,
               show_leaf_counts=True)

    ax1.set_title('Dendrogram (Hierarchical clustering)', fontdict={'fontsize': 25})
    ax1.tick_params(labelsize=18)
    ax1.set_xticks([])
    fig1.tight_layout()
    fig1.savefig('dendrogram.png', dpi=1200)

    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               ax=ax2,
               truncate_mode='lastp',  # show only the last p merged clusters
               show_leaf_counts=True)

    ax2.set_title('Dendrogram (truncated) (Hierarchical clustering)', fontdict={'fontsize': 25})
    ax2.tick_params(labelsize=18)
    fig2.tight_layout()
    fig2.savefig('dendrogram_truncated.png', dpi=1200)


# plot predicted clusters over validation data using trained model

def plot_predicted_clusters_over_validation_data(data_set_to_validate,
                                                 predicted_labels_of_validation_data_set,
                                                 no_of_clusters):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(19.20, 10.80))
    ax1.set_title('Validation over unseen data', fontdict={'fontsize': 25})
    for i in range(len(feature_names)):
        ax1.plot(data_set_to_validate[:, i], color=color_list[i], label=feature_names[i])
        ax1.legend(loc=1, fontsize=14)

    ax1.set_xlabel('Time', fontsize=18)
    ax1.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False, labelsize=18)
    ax1.grid(True)

    ax2.plot(predicted_labels_of_validation_data_set, label='Cluster Index')
    ax2.set_yticks([*range(0, no_of_clusters, 1)])
    ax2.legend(loc=1, fontsize=14)
    ax2.tick_params(labelsize=18)
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig('Validation over unseen data.png', dpi=1200)


def plot_silhouette_avg_vs_clusters(silhouette_avg_list, no_of_cluster_list):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
    ax.plot(no_of_cluster_list, silhouette_avg_list, color='green', marker='o',
            markerfacecolor='blue', markersize=12)
    ax.set_title('Silhouette Analysis', fontdict={'fontsize': 25})
    ax.set_xlabel('Clusters', fontsize=22)
    ax.set_ylabel('Overall average silhouette width', fontsize=22)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig('Overall average silhouette width Vs Clusters.png', dpi=1200)


def plot_predicted_clusters_over_each_feature_in_validation_data(data_set_to_validate,
                                                                 predicted_labels_of_validation_data_set,
                                                                 no_of_clusters):
    for i in range(len(feature_names)):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(19.20, 10.80))
        ax1.plot(data_set_to_validate[:, i], color=color_list[i], label=feature_names[i])
        ax1.set_xlabel('Time', fontsize=18)
        ax1.set_ylabel(feature_names[i], fontsize=18)
        ax1.tick_params(top=True, bottom=False,
                        labeltop=True, labelbottom=False, labelsize=18)
        ax1.set_title(feature_names[i] + ' Vs Predicted Clusters', fontdict={'fontsize': 25})
        ax1.grid(True)
        ax1.tick_params(labelsize=18)

        ax2.plot(predicted_labels_of_validation_data_set, label='Cluster Index')
        ax2.set_yticks([*range(0, no_of_clusters, 1)])
        ax2.legend(loc=1, fontsize=14)
        ax2.tick_params(labelsize=18)
        ax2.grid(True)
        fig.tight_layout()
        fig.savefig(feature_names[i] + ' Vs Predicted Clusters' + '.png', dpi=1200)


def comparison_between_defined_manoeuvres_and_predicted_labels(som_model, data, labels):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
    ax.tick_params(labelsize=18)
    ax.set_aspect('equal')
    xx, yy = som_model.get_euclidean_coordinates()
    umatrix = som_model.distance_map()
    weights = som_model.get_weights()

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
            hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95 / np.sqrt(3),
                                 facecolor=cm.Blues(umatrix[i, j]), alpha=.4, edgecolor='gray')
            ax.add_patch(hex)

    markers = ['*', 'D', '+', 's', '<', '>', 'p']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    for cnt, x in enumerate(data):
        w = som_model.winner(x)  # getting the winner
        # place a marker on the winning position for the sample xx
        wx, wy = som_model.convert_map_to_euclidean(w)
        wy = wy * 2 / np.sqrt(3) * 3 / 4
        plt.plot(wx, wy, markers[labels[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[labels[cnt]], markersize=14, markeredgewidth=2)

    xrange = np.arange(weights.shape[0])
    yrange = np.arange(weights.shape[1])
    plt.xticks(xrange - .5, xrange)
    plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,
                                orientation='vertical', alpha=.4)
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('Distance from nodes in the neighbourhood',
                      fontsize=18)
    cb1.ax.tick_params(labelsize=18)
    plt.gcf().add_axes(ax_cb)

    legend_elements = [Line2D([0], [0], marker=markers[0], color='C0', label='Stop_driving',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker=markers[1], color='C1', label='Start_driving',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker=markers[2], color='C2', label='Accelerate_in_the_lane',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker=markers[3], color='C3', label='Decelerate_in_the_lane',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker=markers[4], color='C4', label='Drive_with_constant_speed',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker=markers[5], color='C5', label='Lane_change_to_the_left',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker=markers[6], color='C6', label='Lane_change_to_the_right',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)
                       ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left',
              borderaxespad=0., ncol=3, fontsize=14)

    fig.tight_layout()
    fig.savefig('Comparision between predicted labels and defined labels.png', dpi=1200)


def plot_hexagonal_topology(som_model, data, labels, no_of_clusters, validation_data_flag):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
    ax.tick_params(labelsize=18)
    ax.set_aspect('equal')
    xx, yy = som_model.get_euclidean_coordinates()
    umatrix = som_model.distance_map()
    weights = som_model.get_weights()

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
            hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95 / np.sqrt(3),
                                 facecolor=cm.Blues(umatrix[i, j]), alpha=.4, edgecolor='gray')
            ax.add_patch(hex)

    if 3 == no_of_clusters:
        if validation_data_flag:
            markers = ['o', '+', 'x']
            colors = ['C0', 'C6', 'C2']
        else:
            markers = ['*', 'D', '+']
            colors = ['C4', 'C5', 'C6']
    elif 4 == no_of_clusters:
        if validation_data_flag:
            markers = ['o', '+', 'x', 's']
            colors = ['C0', 'C7', 'C2', 'C3']
        else:
            markers = ['*', 'D', '+', 's']
            colors = ['C4', 'C5', 'C6', 'C8']
    elif 5 == no_of_clusters:
        if validation_data_flag:
            markers = ['o', '+', 'x', 's', '>']
            colors = ['C0', 'C7', 'C2', 'C3', 'C1']
        else:
            markers = ['*', 'D', '+', 's', '<']
            colors = ['C4', 'C5', 'C6', 'C8', 'C1']

    for cnt, x in enumerate(data):
        w = som_model.winner(x)  # getting the winner
        # place a marker on the winning position for the sample xx
        wx, wy = som_model.convert_map_to_euclidean(w)
        wy = wy * 2 / np.sqrt(3) * 3 / 4
        plt.plot(wx, wy, markers[labels[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[labels[cnt]], markersize=14, markeredgewidth=2)

    xrange = np.arange(weights.shape[0])
    yrange = np.arange(weights.shape[1])
    plt.xticks(xrange - .5, xrange)
    plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues,
                                orientation='vertical', alpha=.4)
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('Distance from nodes in the neighbourhood',
                      fontsize=18)
    cb1.ax.tick_params(labelsize=18)
    plt.gcf().add_axes(ax_cb)

    if 3 == no_of_clusters:
        if validation_data_flag:
            legend_elements = [Line2D([0], [0], marker='o', color='C0', label='0',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='+', color='C6', label='1',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='x', color='C2', label='2',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)]
        else:
            legend_elements = [Line2D([0], [0], marker='*', color='C4', label='0',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='D', color='C5', label='1',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='+', color='C6', label='2',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)]
    elif 4 == no_of_clusters:
        if validation_data_flag:
            legend_elements = [Line2D([0], [0], marker='o', color='C0', label='0',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='+', color='C7', label='1',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='x', color='C2', label='2',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='s', color='C3', label='3',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)
                               ]
        else:
            legend_elements = [Line2D([0], [0], marker='*', color='C4', label='0',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='D', color='C5', label='1',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='+', color='C6', label='2',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='s', color='C8', label='3',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)
                               ]
    elif 5 == no_of_clusters:
        if validation_data_flag:
            legend_elements = [Line2D([0], [0], marker='o', color='C0', label='0',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='+', color='C7', label='1',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='x', color='C2', label='2',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='s', color='C3', label='3',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='>', color='C1', label='4',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)
                               ]
        else:
            legend_elements = [Line2D([0], [0], marker='*', color='C4', label='0',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='D', color='C5', label='1',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='+', color='C6', label='2',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='s', color='C8', label='3',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                               Line2D([0], [0], marker='<', color='C1', label='4',
                                      markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)
                               ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left',
              borderaxespad=0., ncol=3, fontsize=14)

    fig.tight_layout()
    if validation_data_flag:
        fig.savefig('Hexagonal Map Unseen Data.png', dpi=1200)
    else:
        fig.savefig('Hexagonal Map Training Data.png', dpi=1200)


def plot_quantization_topographic_error(som_model, data, no_of_iteration):
    q_error = []
    t_error = []
    iter_x = []
    for i in range(no_of_iteration):
        percent = 100 * (i + 1) / no_of_iteration
        rand_i = np.random.randint(len(data))  # This corresponds to train_random() method.
        som_model.update(data[rand_i], som_model.winner(data[rand_i]), i, no_of_iteration)
        if (i + 1) % 100 == 0:
            q_error.append(som_model.quantization_error(data))
            t_error.append(som_model.topographic_error(data))
            iter_x.append(i)
            sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}%')

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(19.20, 10.80))

    ax1.plot(iter_x, q_error)
    ax1.set_ylabel('Quantization error', fontsize=25)
    ax1.set_xlabel('Iteration index', fontsize=25)
    ax1.set_title('Quantization Error', fontdict={'fontsize': 25})
    ax1.tick_params(labelsize=18)
    ax1.grid(True)

    ax2.plot(iter_x, t_error)
    ax2.set_ylabel('Topographic error', fontsize=25)
    ax2.set_xlabel('Iteration index', fontsize=25)
    ax2.set_title('Topographic Error', fontdict={'fontsize': 25})
    ax2.tick_params(labelsize=18)
    ax2.grid(True)

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('Quantization error.png', dpi=1200)
    fig2.savefig('Topographic error.png', dpi=1200)
