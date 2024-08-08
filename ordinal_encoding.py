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


ordinal_columns = list()
train_data, value_mappings = ordinal_encoding(train_data, ordinal_columns)
valid_data, _ = ordinal_encoding(valid_data, ordinal_columns, value_mappings)
test_data, _ = ordinal_encoding(test_data, ordinal_columns, value_mappings)