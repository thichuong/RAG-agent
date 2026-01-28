---
trigger: always_on
glob: "**/*.py"
description: Mandatory Python development rules including environment, workflow, and code quality.
---

# Python Development Rules

## 1. Environment Management (Strict)
- **Always use Virtual Environment**: All commands, scripts, and installations MUST be executed within the active `.venv`.
- **No Global Installs**: Do not install packages globally. Always ensure the virtual environment is active before running `pip install`.
- **Command Prefix**: If not in an interactive shell where `.venv` is auto-activated, strictly use `.venv/bin/python` or `.venv/bin/pip` to ensure compliance.

## 2. Workflow Adherence
- **Follow Defined Workflows**: You MUST strictly adhere to the steps defined in `.agent/workflows/python.md`.
- **Pre-Commit Checks**: Before declaring a task complete, you must run the analysis and testing steps outlined in the workflow:
  - **Linting**: Run `flake8` to ensure code style and catch errors.
  - **Type Checking**: Run `mypy` to verify type safety.
  - **Testing**: Run `pytest` to verify functionality.

## 3. Code Quality & Standards
- **Type Hints**: All new function definitions should include type hints.
- **Documentation**: Add docstrings to all major functions and classes.
- **Imports**: Remove unused imports. Sort imports according to standard conventions (standard lib -> 3rd party -> local).

## 4. Dependency Management
- **requirements.txt**: Any new package installation must be immediately reflected in `requirements.txt` via `pip freeze > requirements.txt` (or manual addition if pinning specific versions).
- **Clean Environment**: Do not leave unused packages in `requirements.txt`.

## 5. Architecture
- **Keep it Modular**: Follow the existing structure (`src/` for logic, `main.py` for entry).
- **Update Diagrams**: If the project structure changes significantly, update `ARCHITECTURE.md`.