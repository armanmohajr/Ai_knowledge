from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(timeout=10)

# try:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Respond only in JSON."},
#             {"role": "user", "content": "Give me a startup idea in JSON format."}
#         ],
#         response_format={"type": "json_object"},
#     )

#     data = json.loads(response.choices[0].message.content)
#     print(data)

# except Exception as e:
#     print("Error:", e)


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You generate startup ideas in JSON."},
        {"role": "user", "content": "Give me a startup idea in JSON format."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "startup_idea",
            "schema": {
                "type": "object",
                "properties": {
                    "idea_name": {"type": "string"},
                    "problem": {"type": "string"},
                    "solution": {"type": "string"},
                    "target_market": {"type": "string"}
                },
                "required": ["idea_name", "problem", "solution"]
            }
        }
    }
)

print(response.choices[0].message.content)    