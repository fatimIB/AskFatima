from pathlib import Path
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from backend.app.rag.embeddings import get_embedding_model
from backend.app.rag.loader import load_documents
from backend.app.rag.chunker import chunk_documents


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

# Number of candidates produced by RRF
RRF_TOP_K = 10

# Number of final chunks returned after cross-encoder reranking
FINAL_TOP_K = 3

# RRF constant
RRF_K = 60

# Cross-encoder model
MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


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
# 7. Cross-encoder setup
# ---------------------------------------------------------

model = CrossEncoder(MODEL_NAME)


# ---------------------------------------------------------
# 8. Reciprocal Rank Fusion
# ---------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list],
    k: int = RRF_K,
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
# 9. Hybrid RRF candidate retrieval
# ---------------------------------------------------------

def retrieve_rrf_candidates(
    question: str,
    top_k: int = RRF_TOP_K,
):
    """
    Retrieve candidates using:

        Dense retrieval
        +
        BM25 retrieval
        +
        RRF

    Returns the top RRF candidates.

    For this experiment, top_k = 10.
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
    # Keep only top 10 RRF candidates
    # -----------------------------------------------------

    return fused_results[:top_k]


# ---------------------------------------------------------
# 10. Cross-encoder reranking
# ---------------------------------------------------------

def rerank_with_cross_encoder(
    question: str,
    candidates: list,
    top_k: int = FINAL_TOP_K,
):
    """
    Rerank the RRF candidates using the cross-encoder.

    The cross-encoder receives only the RRF candidate pool.

    Example:

        RRF top 10
            ↓
        Cross-Encoder
            ↓
        Final top 3

    Higher cross-encoder score = more relevant.
    """

    if not candidates:
        return []

    # -----------------------------------------------------
    # Create (question, document) pairs
    # -----------------------------------------------------

    pairs = [
        (
            question,
            doc.page_content,
        )
        for doc, rrf_score in candidates
    ]

    # -----------------------------------------------------
    # Score candidates
    # -----------------------------------------------------

    scores = model.predict(pairs)

    # -----------------------------------------------------
    # Combine documents with cross-encoder scores
    # -----------------------------------------------------

    reranked_results = []

    for (
        (doc, rrf_score),
        cross_encoder_score,
    ) in zip(
        candidates,
        scores,
    ):

        reranked_results.append(
            (
                doc,
                float(cross_encoder_score),
                rrf_score,
            )
        )

    # -----------------------------------------------------
    # Sort by cross-encoder score
    # -----------------------------------------------------

    reranked_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # -----------------------------------------------------
    # Return final top 3
    # -----------------------------------------------------

    return reranked_results[:top_k]


# ---------------------------------------------------------
# 11. Final retrieval function
# ---------------------------------------------------------

def retrieve(
    question: str,
    k: int = FINAL_TOP_K,
):
    """
    Complete Hybrid RRF + Cross-Encoder pipeline.

        Dense
          +
        BM25
          ↓
        RRF
          ↓
        Top 10
          ↓
        Cross-Encoder
          ↓
        Top 3

    Returns:

        [
            (Document, cross_encoder_score),
            ...
        ]

    This keeps the same interface as your other
    retrievers so the rest of the RAG pipeline
    does not need to change.
    """

    # -----------------------------------------------------
    # Step 1: RRF top 10
    # -----------------------------------------------------

    rrf_candidates = retrieve_rrf_candidates(
        question,
        top_k=RRF_TOP_K,
    )

    # -----------------------------------------------------
    # Step 2: Cross-encoder reranking
    # -----------------------------------------------------

    reranked_results = rerank_with_cross_encoder(
        question,
        rrf_candidates,
        top_k=k,
    )

    # -----------------------------------------------------
    # Return the same interface expected by pipeline.py
    # -----------------------------------------------------

    return [
        (
            doc,
            cross_encoder_score,
        )
        for (
            doc,
            cross_encoder_score,
            rrf_score,
        ) in reranked_results
    ]


# ---------------------------------------------------------
# 12. Manual testing
# ---------------------------------------------------------

if __name__ == "__main__":

    question = input(
        "Question: "
    ).strip()

    # -----------------------------------------------------
    # RRF top 10
    # -----------------------------------------------------

    rrf_candidates = retrieve_rrf_candidates(
        question,
        top_k=RRF_TOP_K,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HYBRID RRF TOP 10"
    )

    print(
        "=" * 70
    )

    for i, (
        doc,
        rrf_score,
    ) in enumerate(
        rrf_candidates,
        start=1,
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"RRF Rank: {i}"
        )

        print(
            f"RRF Score: {rrf_score:.6f}"
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

    # -----------------------------------------------------
    # Cross-encoder reranking
    # -----------------------------------------------------

    final_results = rerank_with_cross_encoder(
        question,
        rrf_candidates,
        top_k=FINAL_TOP_K,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CROSS-ENCODER FINAL TOP 3"
    )

    print(
        "=" * 70
    )

    for i, (
        doc,
        cross_encoder_score,
        rrf_score,
    ) in enumerate(
        final_results,
        start=1,
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Final Rank: {i}"
        )

        print(
            f"Cross-Encoder Score: "
            f"{cross_encoder_score:.4f}"
        )

        print(
            f"Original RRF Score: "
            f"{rrf_score:.6f}"
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

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Pipeline:"
    )

    print(
        "Dense + BM25 → RRF Top 10 "
        "→ Cross-Encoder → Final Top 3"
    )

    print(
        "=" * 70
    )