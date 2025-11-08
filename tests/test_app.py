from fastapi.testclient import TestClient
from app import app

# Create a TestClient instance
client = TestClient(app)

def test_predict_endpoint():
    """ 
    Tests the /predict endpoint for a successful response and correct structure.
    """
    
    # Example payload that matches the Pydantic model
    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }
    
    # Make a POST request to the /predict endpoint
    response = client.post("/predict", json=payload)
    
    # Assert that the request was successful
    assert response.status_code == 200
    
    # Assert that the response JSON has the expected keys
    response_json = response.json()
    assert "prediction" in response_json
    assert "probability_N" in response_json
    assert "probability_Y" in response_json
    print("✅ API test passed!")
    
def test_root_endpoint():
    """ 
    Tests the root endpoint for a successful response.
    """
    
    # Make a GET request to the root endpoint
    response = client.get("/")
    
    # Assert that the request was successful
    assert response.status_code == 200
    
    # Assert that the response JSON has the expected message
    response_json = response.json()
    assert response_json == {"message": "Welcome to the Loan Approval Prediction API!"}
    print("✅ Root endpoint test passed!")