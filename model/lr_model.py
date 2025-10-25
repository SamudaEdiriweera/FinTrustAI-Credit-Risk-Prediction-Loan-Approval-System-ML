""""
    here, we train the LogisticRegression model
"""

import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report



DIR =os.getcwd()
DATA_PATH = os.path.join(DIR, "../data", "Load_Data.csv")

df = pd.read_csv(DATA_PATH)

# Drop the Load_ID column
df = df.drop(columns=['Loan_ID'])

# Find out all columns with unknown values
for col in df.columns:
    print(f"{col}: {df[col].unique()[ :30]}")

print("_"*50)

# Convert 3+ to numeric
df["Dependents"] = df["Dependents"].replace("3+", "3").astype("object")  # keep '0','1','2','3' as strings

# Column groups by name
numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
binary_cols = ["Credit_History"] # impute most_frequent, keep numeric
ordinal_cols = ["Education", "Dependents"] # ordered
ordinal_categories = [
    ["Not Graduate", "Graduate"], # Educaation
    ["0", "1","2","3"],  # Dependents (post '3+' -> '3')
]

nominal_cols = ["Gender", "Married", "Self_Employed", "Property_Area"]

X = df[numeric_cols + binary_cols + ordinal_cols + nominal_cols]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

### Start to take care of missing values and Feature Scalling
# 🧩 1️⃣ SimpleImputer — fills missing values
# LabelEncoder Vs OrdinalEncoder --- > 
# LabelEncoder can encode one column at a time
# But OrdinalEncoder can encode multiple columns at a time
numeric_transformer = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

binary_transformer = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent"))
])

ordinal_transformer = Pipeline([  # Here, While doing the fill missing value the OrdinalEncoder encodes the 
    ("impute", SimpleImputer(strategy="most_frequent")), 
    ("encode", OrdinalEncoder(categories=ordinal_categories))
])

nominal_transformer = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_cols),
    ("binary", binary_transformer, binary_cols),
    ("ordinal", ordinal_transformer, ordinal_cols),
    ("nominal", nominal_transformer, nominal_cols),
])

classifier = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])


# X_train_processed = preprocess.fit_transform(X_train)
# X_test_processed = preprocess.transform(X_test)
# feature_names = preprocess.get_feature_names_out()
# X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)

# # Show full columns without truncation
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 0)        # auto-detect width
# pd.set_option('display.max_colwidth', None)

# print("✅ Transformed X_train sample:")
# print(X_train_df.head(1).to_string(index=False))
# print("\n✅ y_train sample:")
# print(y_train.head(2))

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)[:, 1]
# pred = (proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC :", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))





































# # Taking care of missing value 
# # Type of variable (categorical vs numerical)

# # Categorical Columns
# # Fill missing values with the mode (most frequent value).
# df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
# df['Married'].fillna(df['Married'].mode()[0], inplace=True)
# df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
# df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)



# # Now convert the entire column to integer type
# df['Dependents'] = df['Dependents'].astype(int)


# # Numerical Columns
# # Use Median for LoanAmount bcz loanamout data is skewed data
# df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# # for col in df.columns:
# #     print(f"{col}: {df[col].unique()[ :30]}")

# # print("_"*50)

# X = df.iloc[ : , : -1].values
# Y = df.iloc[ : , -1 ].values

# """ Encoding the categorical data """

# ## Encoding the Independant variable
# """ Before encoding the let's find out what are ordinal column and nominal columns """
# ''' Gender ---> Nominal = One-Hot
#     Married ---> Nominal
#     Dependents ---> Ordinal = Label Encode
#     Education ---> Ordinal
#     Self_Employed ---> Nominal
#     Education ---> Ordinal
#     Property_Area ---> Nominal
    

# '''
# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0,1,3,4,10])],
#     remainder='passthrough'
# )

# X = ct.fit_transform(X)

# le = LabelEncoder()
# Y = le.fit_transform(Y)

# print(X)
# print(Y)



 