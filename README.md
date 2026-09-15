# Enterprise AI Lab

Evaluation and serving platform for enterprise RAG workloads. The project is
structured around explicit ingestion, retrieval, generation, evaluation, and
provider layers so local and cloud models can be compared reproducibly.

## Local development

Install the package with its development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Run the API:

```bash
uvicorn app.main:app --reload
```

The initial scaffold exposes `GET /health`. RAG ingestion, provider adapters,
and benchmark execution are planned in the implementation phases described in
`plan.md`.
