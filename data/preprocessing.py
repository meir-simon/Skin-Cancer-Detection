import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from data.column_names import cat_cols, feature_cols


def preprocess(df_train, df_test, cat_cols):
    feat_col = feature_cols.copy()

    new_cat = []
    for col in cat_cols:

        col = [col] # make it a list 
        encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
        encoder.fit(df_train[col])
        
        new_cat_cols = [f'{col[0]}_{i}' for i in range(len(encoder.get_feature_names_out()))]
        new_cat.extend(new_cat_cols)

        df_train[new_cat_cols] = encoder.transform(df_train[col])
        df_train[new_cat_cols] = df_train[new_cat_cols].astype('category')

        df_test[new_cat_cols] = encoder.transform(df_test[col])
        df_test[new_cat_cols] = df_test[new_cat_cols].astype('category')

        feat_col.remove(col[0])
        feat_col.extend(new_cat_cols)
    
    return df_train, df_test, feat_col, new_cat


def split_by_patients(train_data_frame, target_column='target', patient_column='patient_id', train_size=0.85, drop_columns=True):
    '''
    This function receives a data frame and splits by patients while maintaining the target ratio.
    :param train_data_frame: Training data frame with the patient IDs and targets inside.
    :param target_column: Name of the target column, 'target' is default.
    :param patient_column: Name of the patient column, 'patient_id' is default.
    :param train_size: Percentage of data to become the training set, 0.85 is default.
    :param drop_columns: When True: target and patient columns are dropped.
    :return: A tuple of 4: x_train, y_train, x_test, y_test.
    '''
    targets = train_data_frame[target_column]
    patients = train_data_frame[patient_column]

    if drop_columns:
        train_data_frame.drop(columns=['target', 'patient_id'], inplace=True)

    # Split the data by patients, while keeping the positive cases distributed properly
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    train_idx, test_idx = next(gss.split(train_data_frame, groups=patients, y=targets))
    x_train, x_test = train_data_frame.iloc[train_idx], train_data_frame.iloc[test_idx]
    y_train, y_test = [targets[i] for i in train_idx], [targets[i] for i in test_idx]

    # Print split stats
    original_train_size = train_data_frame.shape[0]
    train_size = x_train.shape[0]
    original_positive_cases = targets.sum()
    train_positive_cases = sum(y_train)
    print(f'Data split: {train_size * 100 / original_train_size}, {100 - (train_size * 100 / original_train_size)}')
    print(f'Positives cases split: {train_positive_cases * 100 / original_positive_cases}, {100 - (train_positive_cases * 100 / original_positive_cases)}')

    return x_train, y_train, x_test, y_test


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


#encode categorical features
def encode_and_concat(df, encoder, column_name):
    """
    Encodes a categorical column using OneHotEncoder and concatenates the encoded columns to the original DataFrame.
    
    :param df: DataFrame containing the column to encode.
    :param encoder: An instance of OneHotEncoder.
    :param column_name: Name of the column to encode.
    :return: A tuple (updated DataFrame, list of new column names).
    """
    encoder.fit(df[[column_name]])
    encoded_array = encoder.transform(df[[column_name]].fillna('missing'))
    new_columns = encoder.get_feature_names_out([column_name])
    encoded_df = pd.DataFrame(encoded_array, columns=new_columns, index=df.index)
    updated_df = pd.concat([df, encoded_df], axis=1).drop(columns=[column_name])
    return updated_df, new_columns


def add_column(df, name, fill):
    """
    Adds a new column to the DataFrame with a specified name and fill value.
    
    :param df: DataFrame to add the column to.
    :param name: Name of the new column.
    :param fill: Value to fill in the new column.
    :return: Updated DataFrame with the new column.
    """
    df[name] = fill
    return df


def sort_df(df, sort_function=lambda x: x):
    """
    Sorts the columns of a DataFrame according to a given sorting function.
    
    :param df: DataFrame to sort.
    :param sort_function: Function used to determine the sorting order of the columns.
    :return: DataFrame with columns sorted according to the sorting function.
    """
    sorted_columns = sorted(df.columns, key=sort_function)
    return df[sorted_columns]


def encode_and_fill_categorical_columns(*dfs, categorical_columns):
    """
    Encodes categorical columns using OneHotEncoder and ensures all DataFrames have the same set of columns.
    Missing columns in some DataFrames are filled with zeros.
    
    :param dfs: DataFrames to process.
    :param categorical_columns: List of column names to encode.
    :return: Tuple of DataFrames with encoded categorical columns and a list of all new column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')

    df_dict = {i: df.copy() for i, df in enumerate(dfs)}

    all_new_columns = set()

    for col in categorical_columns:
        for i in df_dict:
            df_dict[i], new_columns = encode_and_concat(df_dict[i], encoder, col)
            all_new_columns.update(new_columns)

    for i in df_dict:
        for column in all_new_columns:
            if column not in df_dict[i].columns:
                df_dict[i] = add_column(df_dict[i], column, 0)
        df_dict[i] = sort_df(df_dict[i])

    return tuple(df_dict[i] for i in df_dict), list(all_new_columns)

# # Applying the function
# all_dfs, encoded_columns = encode_and_fill_categorical_columns(train_data, test_data, valid_data,
#                                                                categorical_columns=categorical_columns_to_one_hot)
# train_data, test_data, valid_data = all_dfs


# המרת ערכים לפי חשיבות לסדר
def ordinal_encoding(df, ordinal_features, value_mappings=None):
    if value_mappings is None:
        value_mappings = {}

    for feature in ordinal_features:
        if not df[feature].dtype in [np.int64, np.float64] and feature in df.columns:
            if feature in value_mappings:
                value_mapping = value_mappings[feature]
            else:
                positive_counts = df[df[target_column] == 1][feature].value_counts()
                negative_counts = df[df[target_column] == 0][feature].value_counts()
                ratio = (positive_counts / negative_counts).fillna(0)

                sorted_values = ratio.sort_values().index
                value_mapping = {val: idx + 1 for idx, val in enumerate(sorted_values)}
                value_mappings[feature] = value_mapping

            df[feature] = df[feature].map(value_mapping).fillna(0)
    
    return df, value_mappings

# ordinal_columns = list()
# train_data, value_mappings = ordinal_encoding(train_data, ordinal_columns)
# valid_data, _ = ordinal_encoding(valid_data, ordinal_columns, value_mappings)
# test_data, _ = ordinal_encoding(test_data, ordinal_columns, value_mappings)


def normalize_data(train_df, valid_df, test_df, numerical_features):
    scaler = StandardScaler()
    
    numerical_features = [feature for feature in numerical_features if feature in train_df.columns]
    
    if not numerical_features:
        return train_df, valid_df, test_df
    
    for feature in numerical_features:
        try:
            train_df[[feature]] = scaler.fit_transform(train_df[[feature]])
            valid_df[[feature]] = scaler.transform(valid_df[[feature]])
            test_df[[feature]] = scaler.transform(test_df[[feature]])
        except ValueError as e:
            print(f"שגיאת ערך בנרמול העמודה {feature}: {e}")
        except Exception as e:
            print(f"שגיאה לא צפויה בנרמול העמודה {feature}: {e}")
    
    return train_df, valid_df, test_df

# numerical_columns = ['An array of numeric columns or categories converted to numeric']

# train_data, valid_data, test_data = normalize_data(train_data, valid_data, test_data, numerical_columns)
initiate_dataset_10k_0

import pandas as pd
import os.path
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GroupShuffleSplit
import torch


def initialize_dataset_10k_0(CSV_PATH):
    '''
    return two data frames with 10k target == 0, and all the target == 1.
    train have 3/4 target == 1, and 1/4 in the val_df. each patients
    :param CSV_PATH: path to the csv (train)
    :return: (df, df); train and val
    '''

    # reading dataset
    ISIC_df = pd.read_csv(CSV_PATH)
    ISIC_df = ISIC_df[["isic_id", "patient_id", "target"]]
    ISIC_df["img_path"] = ISIC_df["isic_id"].apply(lambda id: f"{os.path.join(IMG_DIR, id)}.jpg")
    ISIC_df.drop(["isic_id"], axis=1, inplace=True)

    # splitting into target == 1 and target == 0
    target_df = ISIC_df[ISIC_df["target"] == 1].reset_index(drop=True)
    untarget_df = ISIC_df[ISIC_df["target"] == 0].reset_index(drop=True)

    def split_according_patients(df, train_size, drop_patients=True):
        X, y, groups = df, df["target"], df["patient_id"]

        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        gss_gen = gss.split(X, y, groups)
        train_index, val_index = next(gss_gen)
        if drop_patients:
            train_df = df.loc[train_index, ["img_path", "target"]].reset_index(drop=True)
            val_df = df.loc[val_index, ["img_path", "target"]].reset_index(drop=True)
        else:
            train_df = df.loc[train_index, ["img_path", "patient_id", "target"]].reset_index(drop=True)
            val_df = df.loc[val_index, ["img_path", "patient_id", "target"]].reset_index(drop=True)

        print(f"len train_df: {len(train_df)}\nlen val_df: {len(val_df)}")
        return train_df, val_df


    train_target_df, val_target_df = split_according_patients(target_df, 0.75, False)

    def take_common_pationts(untarget_df, target_df):
        '''
        takes the patients that in untarget_df and target_df both.
        :param untarget_df: target == 0
        :param target_df: target == 1
        :return: only the patients that in both df
        '''
        target_patients = target_df["patient_id"].unique()
        untarget_df = untarget_df[untarget_df["patient_id"].isin(target_patients)]
        return pd.concat((untarget_df, target_df), axis=0).reset_index(drop=True)

    train_df = take_common_pationts(untarget_df, train_target_df)
    val_df = take_common_pationts(untarget_df, val_target_df)

    # under sampling
    undersample = RandomUnderSampler(sampling_strategy={0: 10000, 1: len(train_target_df)}, random_state=42)
    train_df, _ = undersample.fit_resample(train_df, train_df["target"])
    train_df = train_df.reset_index(drop=True)
    undersample = RandomUnderSampler(sampling_strategy={0: 10000, 1: len(val_target_df)}, random_state=42)
    val_df, _ = undersample.fit_resample(val_df, val_df["target"])
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def get_std_mean():
    '''
    the function to calculate it is here
    :return: (std: list, mean: list)
    '''
    std = [0.1386, 0.1308, 0.1202]
    mean = [0.6298, 0.5126, 0.4097]
    return std, mean
