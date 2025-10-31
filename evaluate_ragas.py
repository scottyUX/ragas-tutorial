import os
from typing import List, Dict

import faiss  # type: ignore
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)

# Some Ragas versions expose these as answer_*; fall back to response_relevancy if import fails.
try:
    from ragas.metrics import answer_relevancy  # type: ignore
except Exception:  # pragma: no cover
    from ragas.metrics import response_relevancy as answer_relevancy  # type: ignore

try:
    from ragas.metrics import answer_correctness  # type: ignore
except Exception:  # pragma: no cover
    # Fallback to factual_correctness if answer_correctness unavailable
    from ragas.metrics import factual_correctness as answer_correctness  # type: ignore


def get_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required env var: {key}")
    return value


def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)  # in-place normalization for cosine via dot
    index.add(embeddings)
    return index


def embed_corpus(client: OpenAI, corpus: List[str], model: str) -> List[List[float]]:
    vectors: List[List[float]] = []
    # Batch for efficiency (small corpus so simple loop is fine)
    for chunk in corpus:
        emb = client.embeddings.create(model=model, input=chunk)
        vectors.append(emb.data[0].embedding)
    return vectors


def retrieve(
    client: OpenAI,
    index: faiss.IndexFlatIP,
    corpus: List[str],
    query: str,
    embed_model: str,
    top_k: int = 3,
) -> List[str]:
    q = client.embeddings.create(model=embed_model, input=query).data[0].embedding
    xq = faiss.vector_to_array(faiss.swig_ptr(q), len(q))  # list-like
    # faiss expects 2D array
    import numpy as np  # local import to keep global deps minimal

    q_vec = np.array([q], dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, top_k)
    result_indices = indices[0]
    return [corpus[i] for i in result_indices if i != -1]


def generate_answer(client: OpenAI, question: str, contexts: List[str], model: str) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    system = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    user = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        f"Answer concisely."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    load_dotenv()

    openai_api_key = get_env("OPENAI_API_KEY")

    client = OpenAI(api_key=openai_api_key)

    embed_model = "text-embedding-3-small"
    gen_model = "gpt-4o-mini"

    # Tiny toy corpus
    corpus = [
        "OpenAI is an AI research company founded in December 2015.",
        "Founders include Sam Altman, Greg Brockman, Ilya Sutskever, and others.",
        "OpenAI created GPT models and researches alignment and safety.",
        "Elon Musk was an early donor and advisor but is not a current founder of record.",
    ]

    # Questions and ground truths for evaluation
    questions = [
        "Who founded OpenAI?",
        "What does OpenAI research?",
    ]
    ground_truths = [
        "OpenAI was founded in 2015 by Sam Altman, Greg Brockman, Ilya Sutskever, and others.",
        "OpenAI researches AI models like GPT and focuses on alignment and safety.",
    ]

    # Build FAISS index
    corpus_vectors = embed_corpus(client, corpus, embed_model)
    index = build_faiss_index([list(map(float, v)) for v in corpus_vectors])

    contexts: List[List[str]] = []
    answers: List[str] = []

    for q in questions:
        ctx = retrieve(client, index, corpus, q, embed_model, top_k=3)
        contexts.append(ctx)
        ans = generate_answer(client, q, ctx, gen_model)
        answers.append(ans)

    # Build evaluation dataset (datasets.Dataset)
    rows: Dict[str, List] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    evaluation_dataset = Dataset.from_dict(rows)

    scores = evaluate(
        evaluation_dataset,
        metrics=[
            answer_correctness,
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ],
    )

    print(rows)
    print(scores)


if __name__ == "__main__":
    main()
