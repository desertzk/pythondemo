import requests
import os

# Use environment variables to store the API key
API_KEY = os.getenv('DEEPSEEK_API_KEY', 'xxxx')
API_URL = 'https://api.deepseek.com/v1/chat/completions'

# Example data to send to the API
data = {
    'model': 'deepseek-chat',  # Specify the model you want to use
    'messages': [
        {
            'role': 'user',  # The role of the message sender (user or system)
            'content': 'show me nvidia stock all information'  # The actual message
        }
    ],
    'temperature': 0.7,  # Optional: Controls randomness in the response
    'max_tokens': 1000,  # Optional: Limits the length of the response
}

# Headers including the API key
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

try:
    # Make the POST request to the DeepSeek API
    response = requests.post(API_URL, json=data, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("Analysis Result:", result)
    else:
        print(f"Failed to analyze text. Status code: {response.status_code}")
        print("Response:", response.text)

except requests.exceptions.RequestException as e:
    print(f"An error occurred while making the request: {e}")