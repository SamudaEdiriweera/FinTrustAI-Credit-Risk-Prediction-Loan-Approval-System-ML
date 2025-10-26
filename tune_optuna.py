import optuna
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.data_preprocessing import load_and_preprocess_data
from src.train import create_preprocessor
from src.model_evaluation import evaluate_model
from src import config

# --- 1. Setup ---
X_train, X_test, y_train,y_test = load_and_preprocess_data()
preprocessor = create_preprocessor()


def objective(trial):
    """
    The objective function for Optuna to optimize.
    A "trial" is a single run of the model with a specific set of hyperparameters.
    """
    
    # Start a nested MLflow run for this specific trial
    with mlflow.start_run(nested=True):
        # 1. Define the hyperparameter search space
        # Optuna suggests a value for each hyperparameter.
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        c_param = trial.suggest_float('C', 1e-4, 1e2, log=True)
        
        # 2. Create the model pipeline with the suggested hyperparameters
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                solver=solver,
                C = c_param,
                max_iter=config.LOGISTIC_REGRESSION_PARAMS['max_iter'],
                random_state=config.RANDOM_STATE
            ))
        ])
        
        # Evaluate the model using cross-validation
        score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=5, scoring='roc_auc')
        roc_auc = score.mean()
        
        # MLflow logging within the trial's nested run
        mlflow.log_params(trial.params)
        mlflow.log_metric("mean_roc_auc_cv", roc_auc)
        
        return roc_auc
    
# --- Main script execution ---
if __name__ == "__main__":
    print("▶️ Starting Optuna study...")
    # Set the experiment for Mlflow
    mlflow.set_experiment("Hyperparameter_Tuning_Comparison")
    
    # Start a parent Mlflow run to encompass the entire tuning study
    with mlflow.start_run(run_name="Optuna_Run") as parent_run: # Give the parent run a name
        mlflow.set_tag("tuning_method", "Optuna")

        study = optuna.create_study(direction='maximize')
        
        # The objective function now handles its own nested runs
        study.optimize(
            objective, # Pass the function directly
            n_trials=config.OPTUNA_PARAMS['n_trials']
        )

        print("✅ Optuna study finished. Logging best results to parent run...")
        # Log the best results to the PARENT run
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)
        print("🏁 Optuna logging complete.")