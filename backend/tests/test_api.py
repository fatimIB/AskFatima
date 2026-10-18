from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.errors import RateLimitError, AIServiceUnavailableError

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.chat.ask",
        lambda question, history: {
            "answer": "Fatima built AskFatima.",
            "sources": [
                {"source": "projects.md", "section": "AskFatima", "score": 0.9}
            ],
        },
    )
    response = client.post(
        "/chat",
        json={"question": "What is AskFatima?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Fatima built AskFatima."
    assert body["sources"][0]["source"] == "projects.md"


def test_empty_question_is_rejected():
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_too_long_question_is_rejected():
    response = client.post("/chat", json={"question": "a" * 1001})
    assert response.status_code == 422


def test_rate_limit_returns_429(monkeypatch):
    def raise_rate_limit(question, history):
        raise RateLimitError()

    monkeypatch.setattr(
        "backend.app.api.chat.ask",
        raise_rate_limit,
    )
    response = client.post("/chat", json={"question": "Hello?"})
    assert response.status_code == 429


def test_ai_service_down_returns_503(monkeypatch):
    def raise_unavailable(question, history):
        raise AIServiceUnavailableError()

    monkeypatch.setattr(
        "backend.app.api.chat.ask",
        raise_unavailable,
    )
    response = client.post("/chat", json={"question": "Hello?"})
    assert response.status_code == 503


def test_unexpected_error_returns_500(monkeypatch):
    def boom(question, history):
        raise ValueError("boom")

    monkeypatch.setattr(
        "backend.app.api.chat.ask",
        boom,
    )
    response = client.post("/chat", json={"question": "Hello?"})
    assert response.status_code == 500
