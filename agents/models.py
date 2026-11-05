import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Main model: Google Gemini (free tier).
gemini = init_chat_model(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    model_provider="google_genai",
    api_key=os.getenv("GEMINI_API_KEY"),
    max_retries=1,  # try again once if Google is busy
    timeout=30,     # give up on one request after 30 seconds
)

# Backup model: Groq (free tier). Used only when Gemini fails.
if os.getenv("GROQ_API_KEY"):
    groq = init_chat_model(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        model_provider="groq",
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=1,
        timeout=30,
    )
    model = gemini.with_fallbacks([groq])
else:
    print("Note: no GROQ_API_KEY in .env, so there is no backup model.")
    model = gemini


if __name__ == "__main__":
    reply = model.invoke("Say hello in five words.")
    print(reply.text)
    # Shows which model actually answered.
    print("(answered by:", reply.response_metadata.get("model_name"), ")")
