from openai import OpenAI

# Replace with your DeepSeek API key and base URL
DEEPSEEK_API_KEY = "your_deepseek_api_key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # Example base URL

# Initialize the OpenAI client with DeepSeek's configuration
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# Make a request (adjust the payload as per DeepSeek's API)
response = client.chat.completions.create(
    model="deepseek-chat",  # Replace with the correct model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

# Print the response
print(response.choices[0].message.content)