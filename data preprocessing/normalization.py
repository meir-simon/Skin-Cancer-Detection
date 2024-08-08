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

numerical_columns = ['An array of numeric columns or categories converted to numeric']

train_data, valid_data, test_data = normalize_data(train_data, valid_data, test_data, numerical_columns)