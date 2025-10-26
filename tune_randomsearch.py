import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_and_preprocess_data
from src.train import create_preprocessor
from src import config

# --- 1. Setup ---
X_train, X_test, y_train, y_test = load_and_preprocess_data()
preprocessor = create_preprocessor()

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=config.LOGISTIC_REGRESSION_PARAMS['max_iter'],
        random_state=config.RANDOM_STATE
    ))
    
])

# --- 2. Run RandomizedSearchCV ---
print("▶️ Starting RandomizedSearchCV...")
mlflow.set_experiment("Hyperparameter_Tuning_Comparison")

with mlflow.start_run(run_name="RandomizedSearchCV_Run"):
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=config.RANDOM_SEARCH_PARAMS,
        n_iter=config.N_ITER_RANDOM_SEARCH, # Correct: Pass n_iter directly here
        cv = 5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train)
    
        # --- 3. Log Results to MLflow ---
    print("✅ RandomizedSearchCV finished. Logging results...")
    mlflow.log_param("tuning_method", "RandomizedSearchCV")
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("best_roc_auc", random_search.best_score_)

    for i, params in enumerate(random_search.cv_results_['params']):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])
            mlflow.log_metric("std_test_score", random_search.cv_results_['std_test_score'][i])

    print("🏁 RandomizedSearchCV logging complete.")
