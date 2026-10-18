import sys
import types

# Replace the real RAG pipeline before the app is imported.
# API tests therefore need no API keys, Chroma database, or model downloads.
fake_pipeline = types.ModuleType("backend.app.rag.pipeline")


def fake_ask(question, chat_history=None):
    return {
        "answer": "fake answer",
        "sources": [],
    }


fake_pipeline.ask = fake_ask
sys.modules["backend.app.rag.pipeline"] = fake_pipeline
