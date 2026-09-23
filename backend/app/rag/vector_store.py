from pathlib import Path
from langchain_chroma import Chroma
from backend.app.rag.loader import load_documents
from backend.app.rag.chunker import chunk_documents
from backend.app.rag.embeddings import get_embedding_model


# Folder where Chroma will store the vectors
DB_PATH = Path(__file__).resolve().parents[2] / "chroma_db"

def create_vector_store():
    """
    Create the Chroma vector database from the documents.
    """

    # 1. Load documents
    documents = load_documents()

    # 2. Chunk them
    chunks = chunk_documents(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    # 3. Load embedding model
    embedding_model = get_embedding_model()

    print("Creating embeddings...")

    # 4. Create and save Chroma database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(DB_PATH),
    )

    print(f"\nVector database saved to:\n{DB_PATH}")

    return vector_store


if __name__ == "__main__":

    create_vector_store()