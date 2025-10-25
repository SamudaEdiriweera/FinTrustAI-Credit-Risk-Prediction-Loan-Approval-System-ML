import pandas as pd
from sklearn.model_selection import train_test_split
from src import config

def load_and_preprocess_data():
    """ Loads data, performs initial cleaning, and splite into train/test sets."""
    df = pd.read_csv(config.DATA_PATH)
    df = df.drop(columns=['Load_ID'])
    df['Dependents'] = df['Dependents'].replace("3+", "3")
    
    # Use column names from config
    feature_cols = config.NUMERIC_COLS + config.BINARY_COLS + config.ORDINAL_COLS + config.NOMINAL_COLS
    
    X = df[feature_cols]
    y = df[config.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

