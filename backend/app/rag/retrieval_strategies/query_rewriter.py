import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from groq import Groq
from groq import APIStatusError

from backend.app.rag.embeddings import get_embedding_model

load_dotenv()

# --- Chroma setup (same DB as baseline) ---
CHROMA_PATH = Path(__file__).resolve().parents[3] / "chroma_db"

embedding_model = get_embedding_model()

vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model,
)

# --- Groq setup for query rewriting ---
# Deliberately kept on Groq, not Gemini: this is a simple, mechanical
# rewriting task that doesn't need Gemini's full quality, and keeping
# it off Gemini preserves that quota entirely for answer generation.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

groq_client = Groq(api_key=GROQ_API_KEY)

REWRITE_SYSTEM_PROMPT = """
You rewrite natural-language questions into short, keyword-focused search queries,
for use in a semantic search system over a small set of documents about a person
named Fatima (her CV, skills, projects, and background).

Rules:
1. Extract only the key concepts and technical terms from the question.
2. Remove filler words, question words, and the person's name/pronouns
   (they add noise to search, since every document is about her).
3. Do not answer the question. Only output the rewritten search query.
4. Keep it short — a few words, not a full sentence.
5. Output ONLY the rewritten query, nothing else — no explanation, no quotes.

Examples:
Question: "Does Fatima have experience with Flask?"
Rewritten: Flask experience

Question: "What backend frameworks does she know?"
Rewritten: backend frameworks

Question: "Where did Fatima study?"
Rewritten: education university studied
"""

MAX_RETRIES = 3


def rewrite_query(question: str) -> str:
    """
    Use Groq to rewrite a natural-language question into a short,
    keyword-focused search query. Falls back to the original question
    if Groq fails after retries.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": f'Question: "{question}"\nRewritten:'},
                ],
                temperature=0,
                max_tokens=50,
            )

            rewritten = response.choices[0].message.content
            return rewritten.strip() if rewritten else question

        except APIStatusError:
            is_last_attempt = attempt == MAX_RETRIES - 1
            if is_last_attempt:
                print(f"  Query rewriting failed after {MAX_RETRIES} attempts, using original question.")
                return question

            time.sleep(2 ** attempt)

    return question


def retrieve(question: str, k: int = 3):
    """
    Retrieve the k most relevant chunks, after first rewriting
    the question into a shorter, keyword-focused search query.
    """

    rewritten_question = rewrite_query(question)

    results = vector_store.similarity_search_with_score(
        query=rewritten_question,
        k=k,
    )

    return results


if __name__ == "__main__":

    question = input("Question: ")

    rewritten = rewrite_query(question)
    print(f"\nRewritten query: \"{rewritten}\"\n")

    results = vector_store.similarity_search_with_score(
        query=rewritten,
        k=3,
    )

    print("Retrieved Chunks\n")

    for i, (doc, score) in enumerate(results, start=1):
        print("=" * 70)
        print(f"Result {i}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Section: {doc.metadata['section']}")
        print()
        print(doc.page_content[:500])
        print()