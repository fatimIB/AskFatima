from pathlib import Path
from langchain_core.documents import Document


def load_documents(documents_path: str | Path = None):

    if documents_path is None:
        documents_path = Path(__file__).resolve().parents[2] / "documents"

    if not documents_path.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {documents_path}"
        )

    documents = []

    for file in documents_path.glob("*.md"):

        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        document = Document(
            page_content=text,
            metadata={
                "source": file.name
            }
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    docs = load_documents()

    print(f"Loaded {len(docs)} documents\n")

    for doc in docs:
        print("=" * 50)
        print("Source:", doc.metadata["source"])
        print(doc.page_content[:200])
        print()