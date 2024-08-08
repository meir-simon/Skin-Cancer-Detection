from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler

# Function for Near Miss under sampling
def near_miss_undersampling(x_train, y_train, sampling_strategy: dict):
    '''
    Balances the training set by reducing the amount of negative cases using the NearMiss algorithm.
    :param x_train: The unbalanced training set.
    :param y_train: The targets of the unbalanced training set.
    :param sampling_strategy: A dictionary that contains the number of cases for each target value. e.g.: {0: 1000, 1: 400}
    :return: The new balanced training set, and it's targets.
    '''
    undersample = NearMiss(version=1, n_neighbors_ver3=3, sampling_strategy=sampling_strategy)
    new_x_train, new_y_train = undersample.fit_resample(x_train, y_train)
    return new_x_train, new_y_train


# Function for K-means under sampling, NOTICE: Takes a very long time to run
def kmeans_undersampling(x_train, y_train, sampling_strategy: dict):
    '''
    Balances the training set by reducing the amount of negative cases using the K Means algorithm.
    :param x_train: The unbalanced training set.
    :param y_train: The targets of the unbalanced training set.
    :param sampling_strategy: A dictionary that contains the number of cases for each target value. e.g.: {0: 1000, 1: 400}
    :return: The new balanced training set, and it's targets.
    '''
    undersample = ClusterCentroids(sampling_strategy=sampling_strategy)
    new_x_train, new_y_train = undersample.fit_resample(x_train, y_train)
    return new_x_train, new_y_train


# Function for random under sampling
def random_undersampling(x_train, y_train, sampling_strategy: dict):
    '''
    Balances the training set by randomly reducing negative cases.
    :param x_train: The unbalanced training set.
    :param y_train: The targets of the unbalanced training set.
    :param sampling_strategy: A dictionary that contains the number of cases for each target value. e.g.: {0: 1000, 1: 400}
    :return: The new balanced training set, and it's targets.
    '''
    undersample = RandomUnderSampler(random_state=42, sampling_strategy = sampling_strategy)
    new_x_train, new_y_train = undersample.fit_resample(x_train, y_train)
    return new_x_train, new_y_train