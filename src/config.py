import os

# --- Directory Paths ---
DIR = os.getcwd()
DATA_PATH = os.path.join(DIR, "data", "Load_Data.csv")

# --- Data & Model Settings ---
TARGET_COLUMN = "Loan_Status"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Feature Engineering ---
# Column groups by name
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
BINARY_COLS = ["Credit_History"]
ORDINAL_COLS = ["Education", "Dependents"]

# Define the order for ordinal encoding
ORDINAL_CATEGORIES = [
    ["Not Graduate", "Graduate"],
    ["0", "1", "2", "3"],        # Dependents (post '3+' -> '3')
]

NOMINAL_COLS = ["Gender", "Married", "Self_Employed", "Property_Area"]

# --- Model Parameters ---
# UPDATE with the absolute best params from comparison study (mlflow ui we can see the parameters there)
LOGISTIC_REGRESSION_PARAMS = {
    # Use actual best values from the GridSearchCV run
    "solver": "saga",
    "C": 0.1,
    "max_iter": 1000,
    "random_state": RANDOM_STATE
}

# --- Hyperparameter Tuning Search spaces ---

# Note: The 'model__' prefix is required for Scikit-learn pipelines
# when defining a search grid for a step named 'model'.

# 1. GridSearchCV: A samll, explicit grid to test every combination
GRID_SEARCH_PARAMS = {
    'model__solver': ['liblinear', 'saga'],
    'model__C' : [0.01, 0.1, 0.1, 10, 100]
} # Total combiantions: 2 * 5 = 10

# 2. RandomizedSearchCV: A distribution to sample from.
from scipy.stats import loguniform

RANDOM_SEARCH_PARAMS = {
    'model__solver' : ['liblinear', 'saga'],
    'model__C': loguniform(1e-2, 1e2), # Sample from a log-uniform distribution
}

# 3. Optuna: Defines the search space for the objective function.
OPTUNA_PARAMS = {
    'n_trials': 10 # Number of trials to run
}

# Add a separate variable for the number of iterations
N_ITER_RANDOM_SEARCH = 10