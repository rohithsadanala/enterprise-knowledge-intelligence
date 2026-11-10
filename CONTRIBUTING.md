# Contributing

Thanks for your interest in Enterprise Knowledge Intelligence. A few notes
for contributors:

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[test]
pytest
```

## Ground rules

- **No secrets in code.** Keys and tokens go in `.env` (see `.env.example`),
  never in source, tests or fixtures.
- **Keep the offline path working.** The fake-embedding + mock-LLM path
  must always run with no API key and no model download — it is what the
  tests and CI exercise. Real-model providers are opt-in.
- **Cite everything.** Any change to generation must keep citations
  attached to every answer; add a test if you touch the citation path.
- **Tests for retrieval changes.** Chunking, ranking and fusion changes
  need a test in `tests/test_chunking.py` or `tests/test_retrieval.py`
  that shows the behaviour before/after.
- **Type hints and docstrings** on all public functions.

## Adding a provider

1. Implement the provider interface in `src/eki/embeddings.py`,
   `src/eki/llm.py` or `src/eki/vector_store.py`.
2. Register it in the corresponding `get_*` factory.
3. Document the env vars in `.env.example` and `README.md`.
4. Add a test that exercises it with a fake/mock stand-in.
