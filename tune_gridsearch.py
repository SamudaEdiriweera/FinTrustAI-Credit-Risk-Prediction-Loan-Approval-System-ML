import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_and_preprocess_data
from src.train import create_preprocessor
from src import config

# --- 1. Setup ---
# Load data and create preprocessor once
X_train, X_test, y_train, y_test = load_and_preprocess_data()
preprocessor = create_preprocessor()

# Define the model pipeline (without the final model parameters)
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=config.LOGISTIC_REGRESSION_PARAMS['max_iter'],
        random_state=config.RANDOM_STATE
    ))
])

# --- 2. Run GridSearchCV ---
print("▶️ Starting GridSearchCV...")
mlflow.set_experiment("Hyperparameter_Tuning_Comparison")

with mlflow.start_run(run_name="GridSearchCV_Run"):
    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config.GRID_SEARCH_PARAMS,
        cv=5, # 5-fold cross-validation
        scoring='roc_auc',
        n_jobs=-1 # Use all available CPU cores
    )
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # --- 3. Log Results to MLflow ---
    print("✅ GridSearchCV finished. Logging results...")
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_roc_auc", grid_search.best_score_)
    
    # Log each trial as a nested run for detailed tracking
    for i, params in enumerate(grid_search.cv_results_['params']):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", grid_search.cv_results_['mean_test_score'][i])
            mlflow.log_metric("std_test_score", grid_search.cv_results_['std_test_score'][i])

    print("🏁 GridSearchCV logging complete.")