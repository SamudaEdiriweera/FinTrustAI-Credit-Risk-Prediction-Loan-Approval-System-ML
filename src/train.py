from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src import config

def create_preprocessor():
    """ Creates the preprocessing pipeline for the features using config""" 
    numeric_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    
    binary_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent"))
    ])

    ordinal_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=config.ORDINAL_CATEGORIES))
    ])

    nominal_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, config.NUMERIC_COLS),
        ("binary", binary_transformer, config.BINARY_COLS),
        ("ordinal", ordinal_transformer, config.ORDINAL_COLS),
        ("nominal", nominal_transformer, config.NOMINAL_COLS),
    ])
    
    return preprocessor

def train_model(X_train, y_train, preprocessor):
    """ Trains the Logistic Regression model using paramters from config"""
    classifier = Pipeline([
        ("preprocess", preprocessor),
        # Use parameters from the config file
        ("model", LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS))
    ])
    classifier.fit(X_train, y_train)
    return classifier