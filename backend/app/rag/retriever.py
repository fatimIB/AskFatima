import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_PATH)

RETRIEVAL_STRATEGY = os.getenv(
    "RETRIEVAL_STRATEGY"
)

if RETRIEVAL_STRATEGY == "retriever_dense":
    from .retrieval_strategies.retriever_dense import retrieve
elif RETRIEVAL_STRATEGY=="query_rewriting" :
    from .retrieval_strategies.query_rewriter import retrieve
elif RETRIEVAL_STRATEGY=="retriever_hybrid" :
    from .retrieval_strategies.retriever_hybrid import retrieve
elif RETRIEVAL_STRATEGY=="retriever_bm25" :
    from .retrieval_strategies.retriever_bm25 import retrieve
elif RETRIEVAL_STRATEGY=="retriever_cross_encoder" :
    from .retrieval_strategies.retriever_cross_encoder import retrieve
elif RETRIEVAL_STRATEGY=="retriever_hybrid_cross_encoder" :
    from .retrieval_strategies.retriever_hybrid_cross_encoder import retrieve

else:
    raise ValueError(
        f"Unknown retrieval strategy: {RETRIEVAL_STRATEGY}"
    )