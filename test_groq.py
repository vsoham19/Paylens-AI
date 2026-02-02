import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"Testing Groq with API Key: {api_key[:10]}...")

try:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Hello, this is a test."}]
    )
    print("Success! Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error occurred: {e}")
