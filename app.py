import joblib
import pandas as pd
from fastapi import FastAPI, Body # <-- Import Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# (Your app initialization and CORS middleware are the same)
app = FastAPI(title="Loan Approval Prediction API", version="1.0")

# For production, be more specific with your origins for better security.
# For local development, "*" is okay.
origins = [
    "http://localhost:3000",
    "http://172.20.10.2:3000", # The origin from your browser error
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = joblib.load("loan_approval_classifier.joblib")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"💥 Error loading model: {e}")
    model = None

# Pydantic model - We can REMOVE the Config class now
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

# Define the valid example that the docs page will use
VALID_EXAMPLE_DATA = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}

# The root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan Approval Prediction API!"}

# --- THE FINAL, CORRECTED PREDICTION ENDPOINT ---
@app.post("/predict")
def predict(
    application: LoanApplication = Body(..., examples={"Valid Applications": VALID_EXAMPLE_DATA})
):
    """ 
    Receives loan application data and returns the loan approval prediction.
    """
    if not model:
        return {"error":"Model is not loaded. Please check server logs." }
    
    # Convert the Pydantic model to a dictionary
    data = application.model_dump()
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame([data])
    
    # Make predictions
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].tolist()
    
    # Return the results
    return {
        "prediction": prediction,
        "probability_N": probability[0],
        "probability_Y": probability[1]
    }
    
