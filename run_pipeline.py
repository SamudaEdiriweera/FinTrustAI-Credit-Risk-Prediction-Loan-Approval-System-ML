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
    """ Main function to run the ML pipeline"""
    print("▶️ Starting the ML pipeline...")
    
    # 1. Load and preprocess data
    print("⏳ Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("⏳ Loading and preprocessing data...")
    
    # 2. Create the preprocessor
    print("⏳ Creating the preprocessing pipeline...")
    preprocessor = create_preprocessor()
    print("✅ Preprocessor created.")
    
    # 3. Train the model
    print("⏳ Training the model...")
    model = train_model(X_train, y_train, preprocessor)
    print("✅ Model training complete.")
    
    # 4. Evaluate the model
    print("⏳ Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    print("✅ Model evaluation complete.")
    
    print("🏁 Pipeline finished successfully!")

if __name__ == "__main__":
    main()
    