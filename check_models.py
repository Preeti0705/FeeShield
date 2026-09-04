"""
Scratch: list all Gemini models available for your API key.
Run: python check_models.py
"""
import os
import google.genai as genai

api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)

print("Models available for your API key:\n")
for m in client.models.list():
    # Only show generative models, skip embeddings
    if "generate" in str(getattr(m, "supported_actions", "") or "").lower() or "generateContent" in str(getattr(m, "supported_methods", "") or "").lower():
        print(f"  {m.name}")
    else:
        print(f"  {m.name}  (no generateContent)")
