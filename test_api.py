import requests
import json

BASE_URL = "http://localhost:8000"

def test_metrics():
    print("\n--- Testing /metrics ---")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_ask():
    print("\n--- Testing /ask ---")
    payload = {"question": "Does Python increase salary?"}
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_predict():
    print("\n--- Testing /predict ---")
    payload = {
        "features": {
            "Rating": 4.0,
            "age": 10,
            "desc_len": 2500,
            "num_comp": 3,
            "job_simp": "data scientist",
            "seniority": "na",
            "job_state": "NY",
            "Industry": "Aerospace & Defense",
            "Sector": "Aerospace & Defense",
            "Type of ownership": "Company - Private",
            "Size": "501 to 1000 employees",
            "Revenue": "$50 to $100 million (USD)",
            "python_yn": 1,
            "R_yn": 0,
            "spark": 0,
            "aws": 0,
            "excel": 1
        }
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    try:
        test_metrics()
        test_ask()
        test_predict()
    except Exception as e:
        print(f"Error: {e}")
