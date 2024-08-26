from pathlib import Path

from data.isic import get_isic_data  # train XGBoost model



root = Path("../data/isic-2024/")

train_path = root / 'train-metadata.csv'
test_path = root / 'test-metadata.csv'
subm_path = root / 'sample_submission.csv'


train_df, test_df = get_isic_data(train_path, test_path)
print(train_df.shape)
print(test_df.shape)
