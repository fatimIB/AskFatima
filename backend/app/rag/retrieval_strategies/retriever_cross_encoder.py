from backend.app.rag.loader import load_documents
from backend.app.rag.chunker import chunk_documents
from sentence_transformers import CrossEncoder

import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# ---------------------------------------------------------
# 1. Load and prepare the chunks
# ---------------------------------------------------------

_documents = load_documents()
_chunks = chunk_documents(_documents)


# ---------------------------------------------------------
# 2. Load the cross-encoder model
# ---------------------------------------------------------

model = CrossEncoder(MODEL_NAME)


# ---------------------------------------------------------
# 3. Retrieval function
# ---------------------------------------------------------

def retrieve(question: str, k: int = 3):
    """
    Rank document chunks according to how relevant they are
    to the question using a cross-encoder.
    """

    # Create (question, chunk) pairs
    pairs = [
        (question, chunk.page_content)
        for chunk in _chunks
    ]

    # Score every question/chunk pair
    scores = model.predict(pairs)

    # Combine chunks with their scores
    ranked_results = sorted(
        zip(_chunks, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_results[:k]


# ---------------------------------------------------------
# 4. Manual test
# ---------------------------------------------------------

if __name__ == "__main__":

    question = input("Question: ")

    results = retrieve(question)

    print("\nCross-Encoder Retrieved Chunks\n")

    for i, (doc, score) in enumerate(results, start=1):

        print("=" * 70)
        print(f"Result {i}")
        print(f"Cross-Encoder Score: {score:.4f}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Section: {doc.metadata['section']}")
        print()
        print(doc.page_content[:500])
        print()