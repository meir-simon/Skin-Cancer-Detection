from sklearn.preprocessing import OneHotEncoder
import pandas as pd

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

# Applying the function
all_dfs, encoded_columns = encode_and_fill_categorical_columns(train_data, test_data, valid_data,
                                                               categorical_columns=categorical_columns_to_one_hot)
train_data, test_data, valid_data = all_dfs
