from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Create OpenAI client
client = OpenAI()

try:
    # Ask user for input
    user_prompt = input("Prompt: ")

    # Send request to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=50,
    )

    # Print response
    print("\nAI Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("Error:", e)