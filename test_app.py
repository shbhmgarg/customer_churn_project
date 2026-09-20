import requests
import json

# Load sample request
with open('sample_request.json') as f:
    payload = json.load(f)

# Call API
response = requests.post('http://localhost:8000/predict', json=payload)
print("Status code:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))