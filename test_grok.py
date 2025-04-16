import os
from dotenv import load_dotenv
from xai_grok_sdk import XAI

# Load API keys from .env file
load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

# Verify API key
if not GROK_API_KEY:
    raise EnvironmentError("GROK_API_KEY is not set. Please check your .env file.")

# Initialize XAI client
client = XAI(api_key=GROK_API_KEY, model="grok-2-1212")

# Test the API with a simple message
messages = [{"role": "user", "content": "What is the capital of France?"}]

try:
    response = client.invoke(messages)
    print("API Response:")
    print(response)
except Exception as e:
    print(f"Error: {e}") 