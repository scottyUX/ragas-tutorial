# Ragas Tutorial Example

A minimal end-to-end example to evaluate a tiny RAG flow using FAISS for retrieval, OpenAI for embeddings and generation, and Ragas for metrics.

## Setup

1) Python deps:

```bash
pip install -r requirements.txt
```

2) Create `.env` with your keys (see `example.env`):

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
python evaluate_ragas.py
```

It will:
- Build a FAISS index over a tiny corpus with OpenAI embeddings
- Retrieve contexts for sample questions
- Generate answers with OpenAI
- Construct a `datasets.Dataset`
- Evaluate with Ragas metrics: `answer_correctness`, `answer_relevancy`, `faithfulness`, `context_precision`, `context_recall`
- Print the raw rows and the scores

## Code reference
- Docs: [Ragas documentation](https://docs.ragas.io/en/stable/)



```python
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,  # falls back to response_relevancy if needed
    faithfulness,
    context_precision,
    context_recall,
)

# evaluation_dataset is a datasets.Dataset with columns:
# question(str), answer(str), contexts(list[str]), ground_truth(str)

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

print(scores)
```

For more on metrics and usage, see the official docs: [Ragas documentation](https://docs.ragas.io/en/stable/).
