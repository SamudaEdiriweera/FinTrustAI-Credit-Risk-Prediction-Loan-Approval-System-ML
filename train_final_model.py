"""This script's only job is to train and 
save the best model. separates my day-to-day 
experimentation (run_pipeline.py) from the process of 
creating a "blessed" production artifact."""

import joblib
import mlflow
from src.data_preprocessing import load_and_preprocess_data
from src.train import create_preprocessor, train_model
from src.model_evaluation import evaluate_model
from src import config

def main():
    """ 
    Trains and saves the final model pipeline using the best hyperparameters
    defines in the config file.
    """
    mlflow.set_experiment("Production_Models")
    
    with mlflow.start_run():
        print("▶️ Training final production model...")
        
        # Log the parameters from config that define this model
        mlflow.log_params(config.LOGISTIC_REGRESSION_PARAMS)
        
        # 1. Load data
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        
        # 2. Create preprocessor
        preprocessor = create_preprocessor()
        
        # 3. Train the model using the final parameters
        # The train_model function will now automatically use the best params fom config
        final_model = train_model(X_train, y_train, preprocessor)
        print("✅ Final model trained.")
        
        # 4. Evaluate and log metrics
        accuracy, roc_auc, _, _ = evaluate_model(final_model, X_test, y_test)
        mlflow.log_metric("final_accuracy", accuracy)
        mlflow.log_metric("final_roc_auc", roc_auc)
        print(f"Final Model Metrics: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}")

        # 5. Save the entire pipeline (preprocessor + model) to a file
        model_path = "loan_approval_classifier.joblib"
        joblib.dump(final_model, model_path)
        print(f"✅ Final model pipeline saved to {model_path}")

        # 6. Log the final model artifact to MLflow for tracking
        mlflow.sklearn.log_model(final_model, "final_loan_approval_model")
        print("✅ Final model logged to MLflow.")

        print("🏁 Pipeline finished successfully!")
        
if __name__ == "__main__":
    main()