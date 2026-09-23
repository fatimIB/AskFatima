from backend.app.rag.loader import load_documents
from backend.app.rag.chunker import chunk_documents
from rank_bm25 import BM25Okapi
import re


# ---------------------------------------------------------
# 1. Load and chunk the documents
# ---------------------------------------------------------

_documents = load_documents()
_chunks = chunk_documents(_documents)


# ---------------------------------------------------------
# 2. Tokenization
# ---------------------------------------------------------

STOPWORDS = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "be", "been", "being",
    "does", "do", "did",
    "has", "have", "had",
    "having",
    "she", "he", "they", "it",
    "her", "his", "their", "its",
    "what", "where", "who", "when", "which", "how",
    "with", "for", "of", "on", "in", "to",
    "and", "or",
    "know", "knows",
    "currently",
    "kind",
    "Fatima", "fatima",
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase tokens and remove stopwords.
    """

    tokens = re.findall(r"\w+", text.lower())

    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    return tokens


# ---------------------------------------------------------
# 3. Build the BM25 index
# ---------------------------------------------------------

_tokenized_chunks = [
    tokenize(chunk.page_content)
    for chunk in _chunks
]

bm25_index = BM25Okapi(_tokenized_chunks)


# ---------------------------------------------------------
# 4. BM25 retrieval
# ---------------------------------------------------------

def retrieve(question: str, k: int = 3):
    """
    Retrieve the top-k chunks using BM25.
    """

    tokenized_question = tokenize(question)

    scores = bm25_index.get_scores(tokenized_question)

    ranked_results = sorted(
        zip(_chunks, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_results[:k]


# ---------------------------------------------------------
# 5. Manual testing
# ---------------------------------------------------------

if __name__ == "__main__":

    question = input("Question: ")

    results = retrieve(question)

    print("\nRetrieved Chunks\n")

    for i, (doc, score) in enumerate(results, start=1):

        print("=" * 70)
        print(f"Result {i}")
        print(f"BM25 Score: {score:.4f}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Section: {doc.metadata['section']}")
        print()
        print(doc.page_content[:500])
        print()

