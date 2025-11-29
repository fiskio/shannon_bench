# Global Directives for Shannon Benchmark Project

## Code Style

- **Indentation**: Use 2 spaces for indentation in all code files (Python, JavaScript, shell scripts, etc.)
- **Whitespace**: Remove all bad whitespace including trailing whitespace, mixed tabs/spaces, and unnecessary blank lines at end of file

## Python Standards

- **Python Version**: Target Python 3.14 and use modern syntax features
- **Type Annotations**: Use comprehensive type annotations for all functions, methods, and variables
- **Docstrings**: Use Google-style docstrings for all modules, classes, and functions. We will browse them using readthedocs.
- **Data Models**: Use Pydantic instead of dataclasses for data modeling

## Testing and Tooling

- **Testing Framework**: Use `pytest` for all tests
- **Test Files**: Test files should have the same name as the module they test with a `_test.py` suffix (e.g., `module.py` → `module_test.py`)
- **Parameterized Tests**: Use parameterized tests (`pytest.mark.parametrize`) to reduce code duplication and improve test coverage. Tests should be parameterized whenever applicable.
- **Coverage**: Calculate and track test coverage
- **Type Checking**: Use `ty` for static type checking
- **Linting**: Use `ruff check` and `ruff format` - code must have no errors. All changes must pass the linting checks using `task lint`.
- **Import Sorting**: Sort imports using `ruff`

## Environment Management

- **Package Manager**: Environment is managed with uv
- **Test Dependencies**: Install pytest with `uv add pytest` in the active environment
- **Running Tests**: Tests should work with `pytest`. Always run tests via `task test`.


## Architecture and Structure

- **SRC Layout**: All source code must reside in `src/<package_name>`.
- **I/O Boundaries**: Isolate I/O operations (file access, network calls) from core business logic. Core logic should be pure functions whenever possible.

## Modern Idioms

- **Path Handling**: Use `etils.epath` for all file path manipulations. Prefer it over `pathlib.Path` and `os.path`.
- **String Formatting**: Use f-strings for string formatting.
- **Type Hints**:
  - Use `X | Y` for unions (Python 3.10+).
  - Prefer abstract base classes from `collections.abc` (e.g., `Sequence`, `Mapping`) for function arguments.
  - Avoid `Any`. If unavoidable, add a `# type: ignore` with a specific error code and a comment explaining why.

## CLI Development

- **Framework**: Use `typer` for all command-line interfaces.

## Error Handling

- **Custom Exceptions**: Create specific exception classes for domain errors. Inherit from a base project exception.
- **No Bare Excepts**: Never catch `Exception` without re-raising or logging with stack trace.

## Version Control

- **Conventional Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages (e.g., `feat: add new filter`, `fix: resolve crash`).
