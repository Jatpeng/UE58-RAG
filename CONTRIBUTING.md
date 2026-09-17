# Contributing

## Development setup

Create a virtual environment and install the development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the test suite before opening a pull request:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Install the optional local-model dependencies only when working on embedding or
reranking:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[semantic]"
```

## Repository safety

- Do not commit Unreal Engine source, generated chunks, indexes, model caches,
  project-private source, or benchmark outputs derived from private data.
- Do not commit `.env` files, access tokens, API keys, or machine-specific
  absolute paths.
- Keep generated artifacts under the existing ignored `data/`, `work/`, or
  `outputs/` locations.
- Add or update tests for behavior changes and keep the lexical-only test path
  independent of model downloads and external services.
