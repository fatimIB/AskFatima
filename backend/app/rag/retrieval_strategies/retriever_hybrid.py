from pathlib import Path
import re

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from backend.app.rag.embeddings import get_embedding_model
from backend.app.rag.loader import load_documents
from backend.app.rag.chunker import chunk_documents


# ---------------------------------------------------------
# 1. Chroma configuration
# ---------------------------------------------------------

CHROMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "chroma_db"
)


# ---------------------------------------------------------
# 2. Dense retriever setup
# ---------------------------------------------------------

embedding_model = get_embedding_model()

vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model,
)


def load_vector_store():
    return vector_store


# ---------------------------------------------------------
# 3. Load and chunk documents for BM25
# ---------------------------------------------------------

_documents = load_documents()

_chunks = chunk_documents(_documents)


# ---------------------------------------------------------
# 4. BM25 tokenization
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
    Convert text into lowercase tokens
    and remove stopwords.
    """

    tokens = re.findall(
        r"\w+",
        text.lower(),
    )

    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    return tokens


# ---------------------------------------------------------
# 5. Build BM25 index
# ---------------------------------------------------------

_tokenized_chunks = [
    tokenize(chunk.page_content)
    for chunk in _chunks
]

bm25_index = BM25Okapi(
    _tokenized_chunks
)


# ---------------------------------------------------------
# 6. Stable chunk identifier
# ---------------------------------------------------------

def chunk_key(doc):
    """
    Create a stable identifier for a chunk.

    We use source + section + content because the
    Document objects returned by Chroma are not the
    same Python objects as the freshly loaded chunks.
    """

    return (
        doc.metadata.get("source", ""),
        doc.metadata.get("section", ""),
        doc.page_content,
    )


# ---------------------------------------------------------
# 7. Reciprocal Rank Fusion
# ---------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list],
    k: int = 60,
):
    """
    Combine multiple ranked lists using Reciprocal
    Rank Fusion (RRF).

    RRF score:

        score(d) = sum(1 / (k + rank))

    where rank starts at 1.

    The actual retrieval scores from dense and BM25
    are NOT compared directly.
    """

    rrf_scores = {}

    documents = {}

    for ranked_list in ranked_lists:

        for rank, doc in enumerate(
            ranked_list,
            start=1,
        ):

            key = chunk_key(doc)

            documents[key] = doc

            score = 1 / (k + rank)

            rrf_scores[key] = (
                rrf_scores.get(key, 0)
                + score
            )

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (
            documents[key],
            score,
        )
        for key, score in ranked_results
    ]


# ---------------------------------------------------------
# 8. Hybrid retrieval
# ---------------------------------------------------------

def retrieve(
    question: str,
    k: int = 3,
):
    """
    Retrieve documents using:

        Dense retrieval
        +
        BM25 retrieval
        +
        Reciprocal Rank Fusion
    """

    # -----------------------------------------------------
    # Dense retrieval
    # -----------------------------------------------------

    dense_results = (
        load_vector_store()
        .similarity_search_with_score(
            query=question,
            k=len(_chunks),
        )
    )

    dense_ranked_documents = [
        doc
        for doc, score in dense_results
    ]


    # -----------------------------------------------------
    # BM25 retrieval
    # -----------------------------------------------------

    tokenized_question = tokenize(
        question
    )

    bm25_scores = (
        bm25_index.get_scores(
            tokenized_question
        )
    )

    bm25_ranked = sorted(
        zip(_chunks, bm25_scores),
        key=lambda item: item[1],
        reverse=True,
    )

    bm25_ranked_documents = [
        doc
        for doc, score in bm25_ranked
    ]


    # -----------------------------------------------------
    # RRF fusion
    # -----------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        [
            dense_ranked_documents,
            bm25_ranked_documents,
        ]
    )


    # -----------------------------------------------------
    # Return top-k
    # -----------------------------------------------------

    return fused_results[:k]


# ---------------------------------------------------------
# 9. Manual testing
# ---------------------------------------------------------

if __name__ == "__main__":

    question = input(
        "Question: "
    )

    results = retrieve(
        question
    )

    print(
        "\nHybrid RRF Retrieved Chunks\n"
    )

    for i, (doc, score) in enumerate(
        results,
        start=1,
    ):

        print("=" * 70)

        print(
            f"Result {i}"
        )

        print(
            f"RRF Score: {score:.6f}"
        )

        print(
            f"Source : "
            f"{doc.metadata['source']}"
        )

        print(
            f"Section: "
            f"{doc.metadata['section']}"
        )

        print()

        print(
            doc.page_content[:500]
        )

        print()