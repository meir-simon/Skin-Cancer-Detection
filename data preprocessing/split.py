import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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
