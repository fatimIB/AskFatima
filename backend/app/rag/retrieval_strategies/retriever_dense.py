from pathlib import Path
from langchain_chroma import Chroma
from backend.app.rag.embeddings import get_embedding_model


CHROMA_PATH = Path(__file__).resolve().parents[3] / "chroma_db"


embedding_model = get_embedding_model()

vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model,
)

def load_vector_store():
    return vector_store


def retrieve(question: str, k: int = 3):
    """
    Retrieve the k most relevant chunks.
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=question,
        k=k,
    )

    return results


if __name__ == "__main__":

    question = input("Question: ")

    results = retrieve(question)

    print("\nRetrieved Chunks\n")

    for i, (doc, score) in enumerate(results, start=1):

        print("=" * 70)
        print(f"Result {i}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Section: {doc.metadata['section']}")
        print()
        print(doc.page_content[:500])
        print()