import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

print("="*50)
print("1. HEALTH CHECK")
print("="*50)

response = requests.get(f"{BASE_URL}/health")
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "="*50)
print("2. TEST FRAUD TRANSACTION (High Amount)")
print("="*50)

fraud_transaction = {
    "TransactionAmt": 5000.0,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 150.0,
    "card4": "visa",
    "card6": "credit",
    "addr1": 315.0,
    "P_emaildomain": "gmail.com"
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=fraud_transaction
)

print(f"Status Code: {response.status_code}")

# Handle error response
if response.status_code != 200:
    print(f"\nERROR OCCURRED:")
    error_detail = response.json()
    print(json.dumps(error_detail, indent=2))
else:
    result = response.json()
    print(f"\nPrediction: {result['prediction']}")
    print(f"Fraud Probability: {result['fraud_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"\nTop Risk Factors:")
    for factor in result['top_risk_factors']:
        print(f"  - {factor['feature']}: {factor['value']:.2f} ({factor['impact']})")

print("\n" + "="*50)
print("3. TEST LEGITIMATE TRANSACTION (Normal Amount)")
print("="*50)

legit_transaction = {
    "TransactionAmt": 50.0,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 150.0,
    "card4": "visa",
    "card6": "debit",
    "addr1": 315.0,
    "P_emaildomain": "yahoo.com"
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=legit_transaction
)

print(f"Status Code: {response.status_code}")

if response.status_code != 200:
    print(f"\nERROR OCCURRED:")
    error_detail = response.json()
    print(json.dumps(error_detail, indent=2))
else:
    result = response.json()
    print(f"\nPrediction: {result['prediction']}")
    print(f"Fraud Probability: {result['fraud_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommended Action: {result['recommended_action']}")

print("\n" + "="*50)
print("4. MODEL METRICS")
print("="*50)

response = requests.get(f"{BASE_URL}/metrics")
print(f"Status Code: {response.status_code}")
print(json.dumps(response.json(), indent=2))

print("\n" + "="*50)
print("API TESTS COMPLETE!")
print("="*50)