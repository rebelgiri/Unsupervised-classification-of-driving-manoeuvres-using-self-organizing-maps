from minisom import MiniSom
from sklearn import preprocessing


# data normalization between (0,1)
def data_set_normalization(data_set):
    data = preprocessing.minmax_scale(data_set)
    return data


def som_training(normalised_data, som_shape, number_of_features, sigma_value, learning_rate, neighborhood_function,
                 no_of_iteration):
    # Initialization and training
    som_model = MiniSom(som_shape[0], som_shape[1], number_of_features, sigma=sigma_value, learning_rate=learning_rate,
                        neighborhood_function=neighborhood_function, random_seed=10)
    som_model.random_weights_init(normalised_data)

    print("Started Training...")
    som_model.train_random(normalised_data, no_of_iteration, verbose=False)
    neighborhood_error = som_model.topographic_error(normalised_data)
    quantization_error = som_model.quantization_error(normalised_data)
    print(quantization_error)
    print(neighborhood_error)

    som_logs = open('Som_logs.txt', 'a')
    print('quantization_error = ', quantization_error, file=som_logs)
    print('neighborhood_error = ', neighborhood_error, file=som_logs)
    som_logs.close()

    return som_model
