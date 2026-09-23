from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.rag.pipeline import ask

from backend.app.errors import (
    RateLimitError,
    AIServiceUnavailableError,
)


router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    chat_history: list[ChatMessage] = Field(
        default_factory=list,
    )


class SourceInfo(BaseModel):
    source: str
    section: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class RateLimitResponse(BaseModel):
    rate_limited: bool
    answer: None
    sources: list
    message: str


@router.post(
    "/chat",
    response_model=ChatResponse | RateLimitResponse,
)
def chat(request: ChatRequest):

    try:

        history = [
            {
                "role": turn.role,
                "content": turn.content,
            }
            for turn in request.chat_history
        ]

        result = ask(
            request.question,
            history,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=[
                SourceInfo(**src)
                for src in result["sources"]
            ],
        )

    except RateLimitError:

        print(
            "RateLimitError caught by /chat. "
            "Returning HTTP 429."
        )

        raise HTTPException(
            status_code=429,
            detail={
                "rate_limited": True,
                "answer": None,
                "sources": [],
                "message": (
                    "You've reached today's usage limit. "
                    "Please try again tomorrow."
                ),
            },
        )
    except AIServiceUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable. Please try again shortly."
        )

    except Exception as e:

        print(
            f"Unhandled /chat error: "
            f"{type(e).__name__}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Something went wrong processing your question."
            ),
        ) from e

