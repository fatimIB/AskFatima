from langchain_huggingface import HuggingFaceEmbeddings


_embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_embedding_model():
    """
    Load and return the embedding model used throughout the project.
    """
    return _embedding_model


if __name__ == "__main__":

    embedding_model = get_embedding_model()

    query = "Does Fatima have experience with Docker?"

    embedding = embedding_model.embed_query(query)

    print(f"Embedding dimension: {len(embedding)}")
    print()
    print("First 10 values:")
    print(embedding[:10])