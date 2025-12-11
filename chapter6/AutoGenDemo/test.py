import os
from dotenv import load_dotenv
load_dotenv()

print(f"API Key: {os.getenv('LLM_API_KEY')[:10]}...")
print(f"Base URL: {os.getenv('LLM_BASE_URL')}")
print(f"Model: {os.getenv('LLM_MODEL_ID')}")