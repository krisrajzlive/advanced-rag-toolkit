"""RAG evaluation via LangSmith's native `evaluate()` API.

Builds (or reuses) a small QA dataset in LangSmith, runs the RAG pipeline as
the `target`, and scores each run with two evaluators:

- `context_recall`: did retrieval surface a chunk containing the expected
  keyword? (cheap, deterministic signal on the retrieval step)
- `answer_correctness`: an LLM-as-judge grader (using our configured
  LangChain model) that scores the generated answer against the reference
  answer, the standard "LLM judge" pattern LangSmith evaluation is built
  around.

Every run and score is uploaded to the LangSmith project configured via
`LANGSMITH_PROJECT` in `.env`, so results are inspectable in the LangSmith UI.
"""

from __future__ import annotations

from typing import Callable

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from src.config import get_langchain_model

DATASET_NAME = "advanced-rag-toolkit-qa"

DEFAULT_EXAMPLES = [
    {
        "inputs": {"question": "What are the three stages of a RAG pipeline?"},
        "outputs": {
            "answer": "Indexing, retrieval, and generation.",
            "expected_keyword": "generation",
        },
    },
    {
        "inputs": {"question": "What does cosine similarity ignore that dot product is sensitive to?"},
        "outputs": {
            "answer": "Cosine similarity ignores vector magnitude, unlike dot product.",
            "expected_keyword": "magnitude",
        },
    },
    {
        "inputs": {"question": "What does recursive retrieval use to link a summary to detailed chunks?"},
        "outputs": {
            "answer": "A hierarchy of summary nodes that link to detailed child chunks.",
            "expected_keyword": "summary",
        },
    },
]


def _ensure_dataset(client: Client, dataset_name: str, examples: list[dict]) -> str:
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[ex["inputs"] for ex in examples],
            outputs=[ex["outputs"] for ex in examples],
            dataset_id=dataset.id,
        )
    return dataset.id


def _context_recall_evaluator(run: Run, example: Example) -> dict:
    keyword = (example.outputs or {}).get("expected_keyword", "")
    contexts = " ".join((run.outputs or {}).get("contexts", [])).lower()
    hit = keyword.lower() in contexts if keyword else False
    return {"key": "context_recall", "score": int(hit)}


def _make_answer_correctness_evaluator(provider: str | None) -> Callable[[Run, Example], dict]:
    judge = get_langchain_model(provider)

    def _evaluator(run: Run, example: Example) -> dict:
        question = (example.inputs or {}).get("question", "")
        reference = (example.outputs or {}).get("answer", "")
        answer = (run.outputs or {}).get("answer", "")

        prompt = (
            "You are grading a RAG system's answer against a reference answer.\n"
            f"Question: {question}\n"
            f"Reference answer: {reference}\n"
            f"Generated answer: {answer}\n\n"
            "Score the generated answer's correctness from 0 to 1 "
            "(1 = fully correct and consistent with the reference, "
            "0 = incorrect or unrelated). Respond with only the number."
        )
        response = judge.invoke(prompt).content.strip()
        try:
            score = max(0.0, min(1.0, float(response)))
        except ValueError:
            score = 0.0
        return {"key": "answer_correctness", "score": score}

    return _evaluator


def evaluate_rag_pipeline(
    query_fn: Callable[[str], dict],
    dataset_name: str = DATASET_NAME,
    examples: list[dict] | None = None,
    provider: str | None = None,
    experiment_prefix: str = "advanced-rag-toolkit",
):
    """Run LangSmith evaluation over the RAG pipeline.

    `query_fn(question) -> {"answer": str, "contexts": list[str]}` is the
    target system under test - typically a query engine / agent from
    `src.indexing`, `src.retrieval`, or `src.agents`.
    """
    client = Client()
    _ensure_dataset(client, dataset_name, examples or DEFAULT_EXAMPLES)

    def target(inputs: dict) -> dict:
        return query_fn(inputs["question"])

    return evaluate(
        target,
        data=dataset_name,
        evaluators=[
            _context_recall_evaluator,
            _make_answer_correctness_evaluator(provider),
        ],
        client=client,
        experiment_prefix=experiment_prefix,
    )
