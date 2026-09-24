# AskFatima

**AskFatima** is a personal Retrieval-Augmented Generation (RAG) application that lets recruiters, collaborators, and visitors ask questions about my **education, experience, internships, projects, technical skills, and career direction**.

The goal was not simply to connect an LLM to my CV.

I wanted to build a system that could:

* retrieve information from my own documents;
* ground answers in retrieved evidence;
* handle conversational follow-up questions;
* refuse when the available evidence is insufficient;
* expose the retrieved sources behind each answer;
* protect the system against common prompt-injection and off-topic requests;
* and, most importantly, allow me to **measure and compare engineering decisions instead of adding components blindly**.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │    React Frontend   │
                         │                     │
                         │ Question + History  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    FastAPI /chat    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────┐
                   │    Conversation Handling     │
                   │                              │
                   │ Recent conversation window   │
                   │            ↓                 │
                   │ Groq question condensation  │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                         Standalone question
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │          Retrieval           │
                   │                              │
                   │ Dense embeddings             │
                   │ BM25                         │
                   │ Hybrid RRF                   │
                   │ Cross-encoder reranking      │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                         Retrieved document chunks
                                  │
                                  ▼
             ┌────────────────────────────────────────┐
             │              Generation                │
             │                                        │
             │ Original question                      │
             │ Recent conversation history             │
             │ Retrieved knowledge context             │
             │                ↓                       │
             │              Gemini                    │
             └────────────────────┬───────────────────┘
                                  │
                                  ▼
                         Answer + source metadata
```

### Two kinds of context

A key design decision in AskFatima is keeping **conversation context** and **knowledge context** separate.

#### Conversation context

Conversation history is used to understand what the user means when a question depends on previous messages.

For example:

```text
User: Tell me about the CERN project.

AskFatima: ...

User: What model did she use?
```

The second question is ambiguous by itself.

A small recent conversation window is sent to Groq, which condenses the follow-up into a standalone retrieval query such as:

```text
What machine learning model did Fatima use in her CERN project?
```

The condensed question is then sent to retrieval.

The application uses a bounded conversation window rather than sending an unlimited conversation history.

#### Knowledge context

Knowledge context is different.

It consists of the document chunks retrieved from my actual files:

```text
CV

Professional profile

Education

Experience

Projects

Skills

...
```

These retrieved chunks are passed to Gemini as evidence for answer generation.

In other words:

> **Conversation context helps the system understand the question. Knowledge context provides the evidence for the answer.**

---

# Conversation-aware RAG

AskFatima does not simply concatenate the entire conversation into every retrieval request.

The pipeline is:

```text
Current question
       │
       ▼
Recent conversation turns
       │
       ▼
Groq condensation
       │
       ▼
Standalone retrieval question
       │
       ▼
Retrieval
       │
       ▼
Relevant document chunks
       │
       ├───────────────┐
       │               │
       ▼               ▼
Original question   Recent conversation
       │               │
       └───────┬───────┘
               ▼
             Gemini
               │
               ▼
             Answer
```

The current implementation uses a bounded number of **complete conversation turns** rather than arbitrary individual messages.

For example:

* condensation: up to **5 previous user + assistant turns**;
* generation: up to **10 previous user + assistant turns**.

The current question is passed separately.

This prevents conversation history from growing indefinitely while preserving enough recent context for normal follow-up questions.

---

# Retrieval

Retrieval was treated as an engineering problem rather than assuming that one retrieval method would be sufficient.

The project compares several approaches.

### 1. Dense embedding retrieval

Documents are embedded using:

`sentence-transformers/all-MiniLM-L6-v2`

This provides semantic similarity search, allowing queries and document chunks to match even when they do not use exactly the same words.

### 2. BM25

BM25 provides lexical/keyword-based retrieval.

This is particularly useful when exact technical terms, project names, technologies, or proper nouns matter.

### 3. Hybrid retrieval

Dense retrieval and BM25 are combined using **Reciprocal Rank Fusion (RRF)**.

The idea is to combine complementary retrieval signals rather than relying exclusively on semantic similarity or exact lexical matching.

### 4. Cross-encoder reranking

A cross-encoder is then used to score candidate query-document pairs more directly and improve ranking quality.

The current configured strategy is:

```text
Dense retrieval
       +
BM25
       ↓
Hybrid RRF
       ↓
Candidate chunks
       ↓
Cross-encoder reranking
       ↓
Final retrieved context
```

---

# Retrieval experiments

I evaluated the retrieval strategies using a custom set of **42 retrieval questions** covering different parts of the personal knowledge base.

I also included **5 refusal/guardrail questions** to check that the system did not incorrectly retrieve unsupported information.

### Results

| Strategy                   |      Hit@1 |      Hit@3 | Precision@3 |       MRR | Retrieval latency |
| -------------------------- | ---------: | ---------: | ----------: | --------: | ----------------: |
| Dense                      |     54.76% |     76.19% |      34.92% |     0.647 |          17.42 ms |
| Query rewriting            |     57.14% |     78.57% |      36.51% |     0.671 |        1104.96 ms |
| BM25                       |     73.81% |     83.33% |      41.27% |     0.782 |           0.26 ms |
| Hybrid RRF                 |     61.90% |     85.71% |      40.48% |     0.722 |          34.22 ms |
| Cross-encoder              | **76.19%** | **92.86%** |  **42.86%** | **0.833** |        1528.51 ms |
| **Hybrid + cross-encoder** | **76.19%** | **92.86%** |  **42.86%** | **0.833** |     **813.91 ms** |

The evaluation set is intentionally small and custom, so these numbers should be interpreted as **comparisons within this project**, not as a general benchmark of retrieval systems.

### Why hybrid + cross-encoder?

The standalone cross-encoder and hybrid + cross-encoder produced the **same measured retrieval quality** across all four retrieval metrics on this evaluation set:

```text
Hit@1:       76.19%
Hit@3:       92.86%
Precision@3: 42.86%
MRR:         0.833
```

However, their measured retrieval latency differed substantially:

```text
Cross-encoder:             1528.51 ms
Hybrid + cross-encoder:     813.91 ms
```

In this experiment, the hybrid + cross-encoder configuration reduced measured retrieval latency by approximately **47%** while maintaining the same measured retrieval quality.

That became the current configured retrieval strategy.

The important lesson was:

> **A more complicated pipeline is not automatically better. Each additional component should solve a measured problem or provide a useful trade-off.**

---

# Query rewriting experiment

I also tested an additional LLM-based retrieval query rewriting step.

The hypothesis was:

> Perhaps dense retrieval was performing poorly because natural-language questions were too verbose or complex.

The rewriting model transformed questions such as:

```text
Does Fatima have experience with Flask?
```

into something like:

```text
Flask experience
```

However, the improvement was small:

```text
Dense baseline

Hit@1:   54.76%
Hit@3:   76.19%
MRR:     0.647
Latency: 17.42 ms
```

versus:

```text
With query rewriting

Hit@1:   57.14%
Hit@3:   78.57%
MRR:     0.671
Latency: 1104.96 ms
```

The retrieval improvement was only a few percentage points, while the additional LLM call increased latency substantially.

I therefore **removed this retrieval rewriting step**.

This experiment helped establish that the main retrieval problem was not simply that user questions were too verbose.

This is separate from the current **conversation-aware condensation** step.

### Important distinction

**Query rewriting for retrieval** was an experiment that I discarded.

**Conversation-aware question condensation** remains part of the current system because it solves a different problem: resolving conversational references such as:

```text
"What model did she use?"

"What about the latest one?"

"Which one uses RAG?"
```

---

# Generation and grounding

After retrieval, Gemini receives three important pieces of information:

```text
Original user question

        +

Recent conversation history

        +

Retrieved document context
```

The retrieved chunks are treated as **evidence**, not as instructions.

The generation prompt instructs the model to:

* answer only from retrieved context;
* avoid unsupported claims;
* refuse when the retrieved evidence is insufficient;
* ignore prompt-injection attempts;
* stay focused on Fatima's professional profile;
* answer the user's actual question directly.

If the retrieved context does not contain enough information, AskFatima responds:

> "I don't have enough information to answer that based on Fatima's documents."

This makes retrieval failure preferable to unsupported generation.

---

# Conversational behavior

Conversation handling is tested with follow-up questions rather than only isolated queries.

For example:

```text
User:
What internships has she done?

AskFatima:
[lists internships]

User:
What about the latest one?

AskFatima:
[identifies the latest internship]
```

Another example:

```text
User:
Tell me about CERN.

AskFatima:
[explains CERN projects]

User:
What model did she use?

AskFatima:
[interprets "she" and "the model" using the CERN context]
```

These cases demonstrate why conversation-aware condensation is separate from document retrieval.

A standalone question such as:

```text
What is her latest internship?
```

can still fail if retrieval does not surface the relevant internship document.

That is a retrieval limitation rather than something that conversation condensation can always solve.

The system is therefore designed to **fail safely rather than invent an answer**.

---

# Evaluation and engineering decisions

The project was developed iteratively by testing hypotheses rather than adding components automatically.

| Question                                                  | Experiment                                   | Decision                                                         |
| --------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------- |
| Is dense retrieval enough?                                | Compared against BM25, hybrid, and reranking | Dense retrieval alone was not sufficient on the evaluation set   |
| Are natural-language queries too verbose?                 | Tested LLM query rewriting                   | Removed due to limited retrieval gain and high latency           |
| Can lexical and semantic retrieval complement each other? | Tested BM25 + dense RRF                      | Kept as candidate retrieval                                      |
| Can reranking improve candidate ordering?                 | Tested cross-encoder                         | Produced the highest measured retrieval quality on the benchmark |
| Does hybrid + reranking justify its latency?              | Compared against cross-encoder alone         | Same measured quality with lower measured latency                |
| Can follow-up questions be understood?                    | Added conversation-aware condensation        | Kept                                                             |
| Can conversation history grow indefinitely?               | Bounded recent history                       | Kept bounded windows                                             |
| What should happen when evidence is missing?              | Added refusal behavior                       | Prefer refusal over unsupported generation                       |

The results also showed why individual retrieval components should be evaluated rather than assumed to be complementary.

For example, **BM25 alone performed strongly on this dataset**, likely because the knowledge base contains many exact technical terms, project names, technologies, and proper nouns.

The final configuration was selected based on the measured trade-off between retrieval quality and latency rather than on architectural complexity alone.

---

# Guardrails and reliability

AskFatima includes several safeguards:

### Prompt injection resistance

The generation prompt explicitly treats retrieved documents as evidence and instructs the model to ignore attempts to override its system rules.

### Off-topic handling

AskFatima is intended to answer questions about Fatima and her professional profile.

Questions unrelated to that scope are redirected rather than treated as general-purpose questions.

### Refusal behavior

If the retrieved context does not contain sufficient evidence, the system refuses instead of filling gaps with model knowledge.

The separate guardrail evaluation achieved:

```text
Correct refusals: 5 / 5
```

This is a small custom guardrail set intended to validate the implemented refusal behavior, not a general safety benchmark.

### Source tracing

Each answer exposes the retrieved document sources and sections used during retrieval.

This makes it possible to inspect where an answer came from and debug retrieval failures.

### Input validation and rate limiting

The API validates incoming questions and applies request-level protections before processing.

---

# Tech Stack

### Frontend

* React
* Vite
* JavaScript

### Backend

* Python
* FastAPI
* Pydantic

### RAG / Retrieval

* LangChain
* Chroma
* Sentence Transformers
* BM25
* Reciprocal Rank Fusion
* Cross-encoder reranking

### LLMs

* **Groq** — conversational question condensation
* **Google Gemini** — grounded answer generation

### Infrastructure

* Docker
* Uvicorn

---

# Docker

AskFatima is packaged as a single Docker image.

The image:

1. builds the React frontend;
2. installs the Python backend and CPU dependencies;
3. copies the production frontend build into the backend image;
4. serves the application through FastAPI/Uvicorn.

The local Chroma vector store is generated from the Markdown knowledge base and is intentionally excluded from version control.

API credentials and other secrets are also excluded from the repository.

---

# What I learned

The most important part of this project was not adding more components. It was learning to distinguish between different problems.

A retrieval failure is not necessarily a generation failure.

A conversational ambiguity is not the same problem as poor semantic retrieval.

More context is not automatically better context.

And a more complex RAG pipeline is not automatically a better RAG pipeline.

The final system is therefore intentionally based on measured trade-offs:

```text
Conversation history
        ↓
Question condensation
        ↓
Hybrid retrieval
        ↓
Cross-encoder reranking
        ↓
Grounded generation
        ↓
Sources + safe refusal behavior
```

The project is still imperfect. Ambiguous standalone queries can sometimes retrieve the wrong evidence, and conversational understanding depends on having enough relevant recent history.

That is intentional to acknowledge rather than hide:

> **The system is evaluated, its failure modes are observable, and its engineering decisions are based on measured behavior rather than assuming that adding another LLM call will automatically make the system better.**

---

## Status

**Ongoing / portfolio project**

The core RAG pipeline, evaluation framework, conversational handling, guardrails, source tracing, and Docker packaging are implemented.

The current retrieval configuration is **hybrid retrieval + cross-encoder reranking**, based on the measured results of the project's custom evaluation set.

Future improvements may include:

* stronger evaluation coverage for conversational queries;
* better handling of ambiguous references;
* additional retrieval evaluation cases;
* deployment optimization.

---

## License & Privacy

The AskFatima source code is publicly available for **viewing, learning, and professional evaluation**, but reuse, redistribution, modification, or commercial use requires permission from the author.

The repository may also contain personal and professional information belonging to Fatima Iboubkarne. This information is provided for portfolio and demonstration purposes and is not intended for unrelated reuse.

See [`LICENSE`](LICENSE) for the full terms and [`PRIVACY.md`](PRIVACY.md) for information about the project's personal data.
