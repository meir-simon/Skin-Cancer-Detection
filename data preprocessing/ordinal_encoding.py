import numpy as np
import pandas as pd

# Convert values based on importance to order
def ordinal_encoding(df, ordinal_features, value_mappings=None):
    """
    Encodes ordinal features in a DataFrame based on their importance.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing features to encode.
    ordinal_features (list): List of feature names to be encoded.
    value_mappings (dict, optional): Dictionary of precomputed value mappings. 
                                      If None, mappings will be computed.

    Returns:
    pandas.DataFrame: DataFrame with encoded ordinal features.
    dict: Updated dictionary of value mappings.
    """
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

if __name__ == "__main__":
    # Example usage
    # Assuming train_data, valid_data, and test_data are pandas DataFrames
    train_data = pd.DataFrame()  # Replace with actual data
    valid_data = pd.DataFrame()  # Replace with actual data
    test_data = pd.DataFrame()   # Replace with actual data
    
    ordinal_columns = list()  # Replace with actual list of ordinal features
    target_column = 'target'  # Replace with the actual target column name
    
    train_data, value_mappings = ordinal_encoding(train_data, ordinal_columns)
    valid_data, _ = ordinal_encoding(valid_data, ordinal_columns, value_mappings)
    test_data, _ = ordinal_encoding(test_data, ordinal_columns, value_mappings)