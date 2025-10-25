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
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE
}