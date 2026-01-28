---
description: Python Project Analysis & Development Workflow
---

# Python Code Analysis & Development

1.  **Environment Check**
    - Ensure you are running in the correct `.venv`.
    - `source .venv/bin/activate` (if not already active).

2.  **Code Analysis**
    - Run `flake8` or `pylint` to check for style/errors.
    - `flake8 src/ main.py`

3.  **Type Checking**
    - Run `mypy` for type safety.
    - `mypy src/`

4.  **Testing**
    - Run unit tests (if available).
    - `pytest tests/` (or `pytest` if standard discovery).

5.  **Refactoring**
    - When changing code, ensure `ARCHITECTURE.md` is updated if the structure changes.
    - Run `pip freeze > requirements.txt` if new packages are added.
