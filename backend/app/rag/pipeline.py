import logging
import os
import time

from dotenv import load_dotenv
from groq import Groq
from groq import APIStatusError

from backend.app.logging_config import setup_logging
from backend.app.rag.retriever import retrieve
from backend.app.rag.generator import generate_answer


setup_logging()

logger = logging.getLogger(__name__)


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

groq_client = Groq(api_key=GROQ_API_KEY)


CONDENSE_SYSTEM_PROMPT = """
You rewrite a follow-up question into a standalone question, using the
recent conversation for context. The result will be used to search a
small set of documents about a person named Fatima.

Rules:

1. If the question is already standalone, output it UNCHANGED.
   Do not rewrite it unnecessarily.

2. If the question depends on prior context, rewrite it to be
   self-contained by replacing references such as "it", "that project",
   "the first one", "there", or "she" with the specific entity or topic
   from the conversation.

3. ALWAYS preserve explicit names and proper nouns from the user's
   question exactly as written, including project names, company names,
   technologies, organizations, and people.

   Examples:
   - "What did she do in the AskFatima?" must keep "AskFatima".
   - "What did she do at FlyRank?" must keep "FlyRank".
   - "Tell me more about CERN." must keep "CERN".

4. Never replace a specific project, company, organization, technology,
   or other named entity with a generic description.

5. Do not add information that is not supported by the conversation.

6. Output ONLY the rewritten question, nothing else — no explanation.
"""

MAX_RETRIES = 3
HISTORY_WINDOW = 5
GENERATION_HISTORY_WINDOW = 10


def condense_question(question: str, chat_history: list[dict]) -> str:
    """
    Turn a follow-up question into a standalone question using recent
    chat history.

    If the question is already standalone, it is returned unchanged.

    If the question depends on previous conversation context, Groq
    rewrites it into a self-contained question for retrieval.

    Falls back to the original question if the Groq request fails.
    """

    if not chat_history:
        return question

    
    recent_history = chat_history[-(HISTORY_WINDOW * 2):]

    history_text = "\n".join(
        f"{'User' if turn['role'] == 'user' else 'AskFatima'}: {turn['content']}"
        for turn in recent_history
    )

    prompt = (
        f"Conversation:\n"
        f"{history_text}\n\n"
        f"Follow-up: {question}\n"
        f"Standalone:"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": CONDENSE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=100,
            )

            condensed = response.choices[0].message.content

            if not condensed:
                return question

            return condensed.strip()

        except APIStatusError:
            is_last_attempt = attempt == MAX_RETRIES - 1

            if is_last_attempt:
                logger.error(
                    "Question condensing failed after %d attempts. "
                    "Using original question.",
                    MAX_RETRIES,
                )
                return question

            time.sleep(2 ** attempt)

    return question


def ask(
    question: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Complete RAG pipeline:

    Question + history
        ↓
    Condense follow-up question if necessary
        ↓
    Fresh retrieval for the current question
        ↓
    Generate answer from those retrieved chunks
    """

    chat_history = chat_history or []

    # Start measuring the complete RAG pipeline.
    total_start = time.perf_counter()

    logger.info("RAG request started")

    # ---------------------------------------------------------
    # Step 1: Condense the question
    # ---------------------------------------------------------

    condensation_start = time.perf_counter()

    standalone_question = condense_question(
        question,
        chat_history,
    )

    condensation_latency = (
        time.perf_counter() - condensation_start
    )

    logger.info(
        "Query condensation completed in %.3fs",
        condensation_latency,
    )

    # ---------------------------------------------------------
    # Step 2: Retrieve fresh chunks
    # ---------------------------------------------------------

    retrieval_start = time.perf_counter()

    results = retrieve(standalone_question)

    retrieval_latency = (
        time.perf_counter() - retrieval_start
    )

    logger.info(
        "Retrieval completed in %.3fs",
        retrieval_latency,
    )

    logger.info(
        "Retrieved %d documents",
        len(results),
    )

    # ---------------------------------------------------------
    # Step 3: Build context
    # ---------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc, _ in results
    )

    # ---------------------------------------------------------
    # Step 4: Generate the answer
    # ---------------------------------------------------------

    generation_start = time.perf_counter()
    recent_history = chat_history[-(GENERATION_HISTORY_WINDOW * 2):]

    answer = generate_answer(
        question,
        context,
        recent_history,
    )

    generation_latency = (
        time.perf_counter() - generation_start
    )

    logger.info(
        "Generation completed in %.3fs",
        generation_latency,
    )

    # ---------------------------------------------------------
    # Step 5: Total pipeline latency
    # ---------------------------------------------------------

    total_latency = (
        time.perf_counter() - total_start
    )

    logger.info(
        "RAG pipeline completed in %.3fs",
        total_latency,
    )

    return {
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata["source"],
                "section": doc.metadata["section"],
                "score": float(score),
            }
            for doc, score in results
        ],
    }


if __name__ == "__main__":

    history = []

    while True:
        question = input("\nQuestion (or 'quit'): ")

        if question.lower() == "quit":
            break

        result = ask(question, history)

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nSources:\n")

        for src in result["sources"]:
            print(
                f"- {src['source']} / "
                f"{src['section']} "
                f"(score: {src['score']:.4f})"
            )

        history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )