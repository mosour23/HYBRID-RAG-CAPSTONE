import os
from dotenv import load_dotenv
from groq import Groq

# 1. قراءة المفتاح السرّي من ملف .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: API Key not found! Please check your .env file.")
else:
    print("🔌 Connecting to Groq Cloud (Llama-3.1-8B)...")
    client = Groq(api_key=api_key)

    # 2. إرسال تحية للنموذج
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an AI assistant helping an engineer verify his thesis setup."},
            {"role": "user", "content": "Reply in one short sentence: Are you online and ready to be integrated into the Hybrid-RAG Capstone system?"}
        ]
    )

    print("\n🤖 Llama-3 Response:")
    print(f"   \"{response.choices[0].message.content}\"")
    print("\n✅ Success! LLM communication established.")