import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq


def get_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )

    # Groq on-demand tier caps total tokens per request (~8000 incl. completion budget).
    # Defaults stay conservative; raise both env vars on Dev/higher tiers if needed.
    requested = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "3072"))
    hard_cap = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS_HARD_CAP", "4096"))
    max_tokens = max(512, min(requested, hard_cap))

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        max_tokens=max_tokens,
    )

    return llm
