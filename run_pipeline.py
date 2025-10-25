''' a script that orchestrates the entire workflow—from data loading
to model evaluation.'''

from src.data_preprocessing import load_and_preprocess_data
from src.train import create_preprocessor, train_model
from src.model_evaluation import evaluate_model
import mlflow
import mlflow.sklearn
from src import config
import joblib # For saving the preprocessor

def main():
    """ Main function to run the ML pipeline with Mlflow tracking"""
    print("▶️ Starting the ML pipeline...")
    
    # Set the experiment name. If it doesn't exist, MLflow creates it.
    mlflow.set_experiment("Loan_Approval_Prediction")
    
    print("▶️ Starting the ML pipeline...")
    
    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters from the config file
        mlflow.log_param("test_size", config.TEST_SIZE)
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_params(config.LOGISTIC_REGRESSION_PARAMS)
        print("✅ Logged parameters to MLflow.")
    
        # 1. Load and preprocess data
        print("⏳ Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        print("⏳ Loading and preprocessing data...")
    
        # 2. Create the preprocessor
        print("⏳ Creating the preprocessing pipeline...")
        preprocessor = create_preprocessor()
        # Save the preprocessor to be used later
        joblib.dump(preprocessor, 'preprocessor.joblib')
        mlflow.log_artifact('preprocessor.joblib') # Log preprocessor as an artifact
        print("✅ Preprocessor created and logged.")
    
        # 3. Train the model
        print("⏳ Training the model...")
        model = train_model(X_train, y_train, preprocessor)
        print("✅ Model training complete.")
    
        # 4. Evaluate the model
        print("⏳ Evaluating the model...")
        accuracy, roc_auc, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
        print("✅ Model evaluation complete.")
        
        # 5. Log metrics to MLflow
        print("⏳ Logging metrics to MLflow...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        # mlflow.log_metric("conf_matrix", conf_matrix)
        # mlflow.log_metric("class_report", class_report)
        print("✅ Metrics logged.")
        
        # 6. Log the trained model to MLflow
        print("⏳ Logging model to MLflow...")
        # This logs the model in a format that MLflow understands
        mlflow.sklearn.log_model(model, "logistic_regresstion_model")
        print("✅ Model logged.")
    
        print("🏁 Pipeline finished successfully!")

if __name__ == "__main__":
    main()
    
"""*Note: We also added `joblib` to save the preprocessor object. 
This is important because for future predictions on new data, 
you'll need to apply the exact same preprocessing steps.
Install it if you haven't already: `pip install joblib` and 
add it to `requirements.txt`.*"""
    