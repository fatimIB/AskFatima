import json
import os
from pathlib import Path
from datetime import datetime
import time
from datetime import datetime
from dotenv import load_dotenv

from backend.app.rag.pipeline import ask
from backend.app.rag.retriever import retrieve

load_dotenv()

RETRIEVAL_STRATEGY = os.getenv(
    "RETRIEVAL_STRATEGY",
    "unknown",
)

EVALUATION_PATH = (
    Path(__file__).parent / "evaluation.json"
)

REPORTS_FOLDER = (
    Path(__file__).parent.parent
    / "evaluation_reports"
    / RETRIEVAL_STRATEGY
)

REPORTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# These phrases are used to detect whether
# the final answer refused an out-of-scope question.
REFUSAL_PHRASES = [
    "don't have enough information",
    "only answer",
    "only answers",
    "cannot fulfill",
    "cannot comply",
    "cannot assist",
    "dedicated solely",
    "unable to answer",
    "i can only",
    "i cannot",
]


def load_test_cases():
    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        test_cases = json.load(file)

    return test_cases


def doc_to_pair(doc):
    return {
        "source": doc.metadata["source"],
        "section": doc.metadata["section"],
    }


def is_expected(pair, expected_pairs):
    """
    A question may have multiple acceptable source chunks.

    If the retrieved chunk matches ANY expected pair,
    it counts as relevant.
    """
    return pair in expected_pairs


def evaluate_retrieval(verbose: bool = True):
    """
    Evaluate retrieval for factual questions.

    A question can have multiple expected sources.
    These are alternative valid sources, not chunks
    that must all be retrieved.

    Therefore:

    Hit@1:
        At least one acceptable source is ranked #1.

    Hit@3:
        At least one acceptable source appears in top 3.

    MRR:
        Reciprocal rank of the first acceptable source.

    Recall@3 and Precision@3:
        Based on whether retrieved chunks are acceptable
        alternatives, rather than requiring all expected
        sources to be retrieved.
    """

    test_cases = load_test_cases()

    hit1_count = 0
    hit3_count = 0
    precision3_sum = 0.0
    mrr_sum = 0.0
    retrieval_latency_sum = 0.0

    total = 0
    failures = []

    for case in test_cases:

        # Refusal questions are evaluated separately.
        if case.get("should_refuse", False):
            continue

        total += 1

        question = case["question"]
        expected_pairs = case["expected"]

        start_time = time.perf_counter()

        results = retrieve(question)

        retrieval_latency = (
            time.perf_counter() - start_time
        )

        retrieval_latency_sum += retrieval_latency

        retrieved_pairs = [
            doc_to_pair(doc)
            for doc, _ in results
        ]

        top3_pairs = retrieved_pairs[:3]

        # --------------------------------------------------
        # Hit@1
        # --------------------------------------------------

        hit1 = (
            len(retrieved_pairs) > 0
            and is_expected(
                retrieved_pairs[0],
                expected_pairs,
            )
        )

        if hit1:
            hit1_count += 1

        # --------------------------------------------------
        # Hit@3
        # --------------------------------------------------

        hit3 = any(
            is_expected(pair, expected_pairs)
            for pair in top3_pairs
        )

        if hit3:
            hit3_count += 1

        # --------------------------------------------------
        # Precision@3
        # --------------------------------------------------
        #
        # Since expected sources are alternatives,
        # a retrieved expected source is relevant.
        #
        # We do NOT divide by the number of expected
        # sources because they are not all required.
        # --------------------------------------------------

        relevant_retrieved = [
            pair
            for pair in top3_pairs
            if is_expected(
                pair,
                expected_pairs,
            )
        ]

        if top3_pairs:
            precision3 = (
                len(relevant_retrieved)
                / len(top3_pairs)
            )
        else:
            precision3 = 0.0

        precision3_sum += precision3

        # --------------------------------------------------
        # MRR
        # --------------------------------------------------

        first_relevant_rank = None

        for rank, pair in enumerate(
            retrieved_pairs,
            start=1,
        ):
            if is_expected(
                pair,
                expected_pairs,
            ):
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            reciprocal_rank = (
                1 / first_relevant_rank
            )
        else:
            reciprocal_rank = 0.0

        mrr_sum += reciprocal_rank

        # --------------------------------------------------
        # Failures
        # --------------------------------------------------

        if not hit1:
            failures.append({
                "question": question,
                "expected": expected_pairs,
                "retrieved": retrieved_pairs,
                "found_in_top3": hit3,
                "hit3": hit3,
                "precision3": precision3,
                "reciprocal_rank": reciprocal_rank,
            })

        # --------------------------------------------------
        # Verbose output
        # --------------------------------------------------

        if verbose:

            if hit1:
                status = "✅"
            elif hit3:
                status = "⚠️"
            else:
                status = "❌"

            print(
                f'{status} "{question}"'
            )

            print("Expected:")
            for pair in expected_pairs:
                print(
                    f"   ✓ {pair}"
                )

            print("Retrieved Top-3:")

            for rank, pair in enumerate(
                top3_pairs,
                start=1,
            ):
                marker = (
                    "✓"
                    if is_expected(
                        pair,
                        expected_pairs,
                    )
                    else " "
                )

                print(
                    f"   {rank}. {pair} {marker}"
                )

            print(
                f"Hit@1: {hit1} | "
                f"Hit@3: {hit3} | "
                f"Precision@3: "
                f"{precision3:.2f} | "
                f"RR: {reciprocal_rank:.3f}"
            )

            print()

    # ------------------------------------------------------
    # Final retrieval metrics
    # ------------------------------------------------------

    if total:
        hit1 = hit1_count / total
        hit3 = hit3_count / total
        precision3 = precision3_sum / total
        mrr = mrr_sum / total
        average_latency = retrieval_latency_sum / total
    else:
        hit1 = 0.0
        hit3 = 0.0
        precision3 = 0.0
        mrr = 0.0
        average_latency = 0.0

    return {
        "hit1": hit1,
        "hit3": hit3,
        "precision3": precision3,
        "mrr": mrr,
        "average_latency": average_latency,
        "hit1_count": hit1_count,
        "hit3_count": hit3_count,
        "total": total,
        "failures": failures,
    }


def evaluate_guardrails(verbose: bool = True):
    """
    Evaluate the five refusal questions.

    These questions are NOT part of retrieval metrics.

    Instead, they are sent through the complete pipeline
    using ask(), and we check whether the final answer
    contains a refusal response.
    """

    test_cases = load_test_cases()

    correct_refusals = 0
    total = 0
    failures = []

    for case in test_cases:

        if not case.get("should_refuse", False):
            continue

        total += 1

        question = case["question"]

        # Use the complete RAG pipeline here because
        # guardrails concern the final answer.
        response = ask(question)

        answer = response["answer"].lower()

        refused = any(
            phrase in answer
            for phrase in REFUSAL_PHRASES
        )

        if refused:
            correct_refusals += 1
        else:
            failures.append({
                "question": question,
                "answer": response["answer"],
            })

        if verbose:

            status = (
                "✅"
                if refused
                else "❌"
            )

            print(
                f'{status} "{question}" '
                f"→ refused: {refused}"
            )

    return {
        "correct_refusals": correct_refusals,
        "total": total,
        "failures": failures,
    }


def save_report(
    retrieval,
    guardrails,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        REPORTS_FOLDER
        / f"report_{timestamp}.json"
    )

    txt_path = (
        REPORTS_FOLDER
        / f"report_{timestamp}.txt"
    )

    report = {
        "retriever": RETRIEVAL_STRATEGY,
        "timestamp": timestamp,

        "retrieval": {
            "questions": retrieval["total"],
            "hit_at_1": retrieval["hit1"],
            "hit_at_3": retrieval["hit3"],
            "precision_at_3": retrieval["precision3"],
            "mrr": retrieval["mrr"],
            "average_latency_seconds": retrieval["average_latency"],
            "hit_at_1_count": retrieval["hit1_count"],
            "hit_at_3_count": retrieval["hit3_count"],
            "failures": retrieval["failures"],
        },

        "guardrails": {
            "correct_refusals": (
                guardrails["correct_refusals"]
            ),
            "total": guardrails["total"],
            "failures": guardrails["failures"],
        },
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "RAG EVALUATION REPORT\n"
        )
        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Retriever: "
            f"{RETRIEVAL_STRATEGY}\n"
        )

        file.write(
            f"Timestamp: {timestamp}\n\n"
        )

        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------

        file.write(
            "Retrieval Evaluation\n"
        )
        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"Questions: "
            f"{retrieval['total']}\n"
        )

        file.write(
            f"Hit@1: "
            f"{retrieval['hit1'] * 100:.2f}%\n"
        )

        file.write(
            f"Hit@3: "
            f"{retrieval['hit3'] * 100:.2f}%\n"
        )

        file.write(
            f"Precision@3: "
            f"{retrieval['precision3'] * 100:.2f}%\n"
        )

        file.write(
            f"MRR: "
            f"{retrieval['mrr']:.3f}\n\n"
        )

        file.write(
            f"Average Retrieval Latency: "
            f"{retrieval['average_latency'] * 1000:.2f} ms\n\n"
        )

        # --------------------------------------------------
        # Guardrails
        # --------------------------------------------------

        file.write(
            "Guardrail Evaluation\n"
        )
        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"Correct Refusals: "
            f"{guardrails['correct_refusals']}"
            f"/{guardrails['total']}\n"
        )

        if guardrails["failures"]:

            file.write(
                "\nGuardrail Failures\n"
            )
            file.write(
                "-" * 60 + "\n"
            )

            for failure in guardrails["failures"]:

                file.write(
                    f'\nQuestion: '
                    f'{failure["question"]}\n'
                )

                file.write(
                    f'Answer: '
                    f'{failure["answer"]}\n'
                )

        # --------------------------------------------------
        # Retrieval failures
        # --------------------------------------------------

        if retrieval["failures"]:

            file.write(
                "\nRetrieval Failures\n"
            )
            file.write(
                "-" * 60 + "\n"
            )

            for failure in retrieval["failures"]:

                file.write(
                    f'\nQuestion: '
                    f'{failure["question"]}\n'
                )

                file.write("Expected:\n")

                for pair in failure["expected"]:

                    file.write(
                        f"   {pair}\n"
                    )

                file.write("Retrieved:\n")

                for i, pair in enumerate(
                    failure["retrieved"],
                    start=1,
                ):

                    file.write(
                        f"   {i}. {pair}\n"
                    )

                file.write(
                    f"Hit@3: "
                    f"{failure['hit3']}\n"
                )

                file.write(
                    f"Precision@3: "
                    f"{failure['precision3']:.2f}\n"
                )

                file.write(
                    f"RR: "
                    f"{failure['reciprocal_rank']:.3f}\n"
                )

    print("\nReports saved:")
    print(json_path)
    print(txt_path)


def print_report():

    print(
        f"\nActive retrieval strategy: "
        f"{RETRIEVAL_STRATEGY}\n"
    )

    # ======================================================
    # RETRIEVAL
    # ======================================================

    print(
        "Running retrieval evaluation...\n"
    )

    retrieval = evaluate_retrieval()

    print()

    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    print()

    print(
        f"Questions          : "
        f"{retrieval['total']}"
    )

    print(
        f"Hit@1              : "
        f"{retrieval['hit1'] * 100:.2f}%"
    )

    print(
        f"Hit@3              : "
        f"{retrieval['hit3'] * 100:.2f}%"
    )

    print(
        f"Precision@3        : "
        f"{retrieval['precision3'] * 100:.2f}%"
    )

    print(
        f"MRR                : "
        f"{retrieval['mrr']:.3f}"
    )

    print(
        f"Avg Retrieval Latency: "
        f"{retrieval['average_latency'] * 1000:.2f} ms"
    )

    # ======================================================
    # RETRIEVAL FAILURES
    # ======================================================

    if retrieval["failures"]:

        near_misses = [
            failure
            for failure in retrieval["failures"]
            if failure["found_in_top3"]
        ]

        total_misses = [
            failure
            for failure in retrieval["failures"]
            if not failure["found_in_top3"]
        ]

        if near_misses:

            print()
            print(
                "Retrieval Near-Misses"
            )
            print(
                "-" * 60
            )

            for failure in near_misses:

                print(
                    f'\nQuestion: '
                    f'"{failure["question"]}"'
                )

                print("Expected:")

                for pair in failure["expected"]:

                    print(
                        f"   ✓ {pair}"
                    )

                print("Retrieved:")

                for i, pair in enumerate(
                    failure["retrieved"][:3],
                    start=1,
                ):

                    marker = (
                        "✓"
                        if pair in failure["expected"]
                        else " "
                    )

                    print(
                        f"   {i}. {pair} "
                        f"{marker}"
                    )

                print(
                    f"Hit@3: "
                    f"{failure['hit3']}"
                )

                print(
                    f"RR: "
                    f"{failure['reciprocal_rank']:.3f}"
                )

        if total_misses:

            print()
            print(
                "Retrieval Total Misses"
            )
            print(
                "-" * 60
            )

            for failure in total_misses:

                print(
                    f'\nQuestion: '
                    f'"{failure["question"]}"'
                )

                print("Expected:")

                for pair in failure["expected"]:

                    print(
                        f"   ✓ {pair}"
                    )

                print("Retrieved:")

                for i, pair in enumerate(
                    failure["retrieved"][:3],
                    start=1,
                ):

                    print(
                        f"   {i}. {pair}"
                    )

                print(
                    f"Hit@3: "
                    f"{failure['hit3']}"
                )

                print(
                    f"RR: "
                    f"{failure['reciprocal_rank']:.3f}"
                )

    # ======================================================
    # GUARDRAILS
    # ======================================================

    print()
    print(
        "Running guardrail evaluation...\n"
    )

    guardrails = evaluate_guardrails()

    print()

    print("=" * 60)
    print("GUARDRAIL EVALUATION")
    print("=" * 60)

    print()

    print(
        f"Correct Refusals    : "
        f"{guardrails['correct_refusals']}"
        f"/{guardrails['total']}"
    )

    if guardrails["total"]:

        guardrail_rate = (
            guardrails["correct_refusals"]
            / guardrails["total"]
            * 100
        )

        print(
            f"Refusal Accuracy   : "
            f"{guardrail_rate:.2f}%"
        )

    if guardrails["failures"]:

        print()
        print(
            "Guardrail Failures"
        )
        print(
            "-" * 60
        )

        for failure in guardrails["failures"]:

            print(
                f'\nQuestion: '
                f'"{failure["question"]}"'
            )

            print(
                f'Answer: '
                f'{failure["answer"]}'
            )

    # ======================================================
    # SAVE
    # ======================================================

    save_report(
        retrieval,
        guardrails,
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    print_report()

