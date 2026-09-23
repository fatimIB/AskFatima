import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.rag.loader import load_documents


# Maximum size of a normal chunk.
# If a section becomes larger than this, we split it further.
MAX_CHUNK_SIZE = 3000

# Fallback splitter for unusually large sections.
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=100,
)


def get_document_category(source: str) -> str:
    """
    Convert the document filename into a human-readable category.

    Examples:
        experience.md   -> Experience
        education.md    -> Education
        projects.md     -> Projects
        skills.md       -> Skills
        certificates.md -> Certificates
        about.md        -> About
    """

    category_map = {
        "about.md": "About",
        "education.md": "Education",
        "experience.md": "Experience",
        "projects.md": "Projects",
        "skills.md": "Skills",
        "certificates.md": "Certificates",
    }

    filename = Path(source).name.lower()

    return category_map.get(
        filename,
        Path(filename).stem.replace("_", " ").title(),
    )


def create_chunk(
    text: str,
    metadata: dict,
    category: str,
) -> Document:
    """
    Create one chunk while adding the parent document category
    to both the text and metadata.

    The category is included in page_content so that both:
    - dense embeddings
    - BM25 keyword search

    can use it during retrieval.
    """

    enriched_text = (
        f"Document category: {category}\n\n"
        f"{text.strip()}"
    )

    chunk_metadata = metadata.copy()
    chunk_metadata["category"] = category

    return Document(
        page_content=enriched_text,
        metadata=chunk_metadata,
    )


def chunk_documents(documents):
    chunks = []

    for document in documents:
        text = document.page_content.strip()

        # ---------------------------------------------------------
        # Determine the parent document category.
        #
        # Example:
        #
        # experience.md -> Experience
        # projects.md   -> Projects
        # ---------------------------------------------------------
        category = get_document_category(
            document.metadata.get("source", "")
        )

        # ---------------------------------------------------------
        # Split before every ## heading.
        #
        # Example:
        #
        # ## Professional Profile
        # ...
        # ## Career Direction
        # ...
        #
        # becomes:
        #
        # [Professional Profile]
        # [Career Direction]
        #
        # The heading stays attached to its content.
        # ---------------------------------------------------------
        sections = re.split(
            r"(?=^## )",
            text,
            flags=re.MULTILINE,
        )

        # ---------------------------------------------------------
        # If there are no ## headings, keep the whole document
        # as one chunk.
        #
        # This is what happens to:
        #   skills.md
        #   certificates.md
        # ---------------------------------------------------------
        if len(sections) == 1:
            section = sections[0].strip()

            if section:
                metadata = document.metadata.copy()

                # Store the document category in metadata.
                metadata["category"] = category

                # Use the # title as the section name.
                first_line = section.splitlines()[0]

                if first_line.startswith("# "):
                    metadata["section"] = (
                        first_line.replace("# ", "").strip()
                    )
                else:
                    metadata["section"] = "Document"

                # Small document -> one chunk
                if len(section) <= MAX_CHUNK_SIZE:
                    chunks.append(
                        create_chunk(
                            text=section,
                            metadata=metadata,
                            category=category,
                        )
                    )

                # Large document -> recursively split it
                else:
                    split_chunks = fallback_splitter.create_documents(
                        texts=[
                            f"Document category: {category}\n\n{section}"
                        ],
                        metadatas=[metadata],
                    )

                    chunks.extend(split_chunks)

            continue

        # ---------------------------------------------------------
        # Documents with ## headings
        # ---------------------------------------------------------
        for section in sections:
            section = section.strip()

            if not section:
                continue

            # The first part is usually just:
            #
            # # About Fatima
            #
            # It is only the document title, so we skip it.
            if not section.startswith("## "):
                continue

            # Extract the ## heading.
            first_line = section.splitlines()[0]

            heading = first_line.replace("## ", "").strip()

            metadata = document.metadata.copy()
            metadata["section"] = heading
            metadata["category"] = category

            # -----------------------------------------------------
            # Normal section -> keep it as one semantic chunk
            # -----------------------------------------------------
            if len(section) <= MAX_CHUNK_SIZE:
                chunks.append(
                    create_chunk(
                        text=section,
                        metadata=metadata,
                        category=category,
                    )
                )

            # -----------------------------------------------------
            # Very large section -> recursive fallback
            # -----------------------------------------------------
            else:
                split_chunks = fallback_splitter.create_documents(
                    texts=[
                        f"Document category: {category}\n\n{section}"
                    ],
                    metadatas=[metadata],
                )

                chunks.extend(split_chunks)

    # Give every chunk a unique ID.
    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks\n")

    for chunk in chunks:
        print("=" * 70)
        print("Chunk ID:", chunk.metadata.get("chunk_id"))
        print("Source:", chunk.metadata.get("source"))
        print("Category:", chunk.metadata.get("category"))
        print("Section:", chunk.metadata.get("section"))
        print()
        print(chunk.page_content)
        print()