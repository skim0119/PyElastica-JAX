# AGENTS

- For numeric precondition checks, prefer `assert` over `if ...: raise ...` patterns.
- Use explicit assertion messages so failures explain the violated invariant directly.

- README.md is the main documentation for building simulation. Take a look if you are not sure how to use this package.
- Don't stage/commit .gitignore.

## Style

- Do not import-guard JAX: it will raise ImportError if necessary.
- `del kwargs` or del unused parameters/variables is not necessary.

## Short commands

- `inline` means copy the code inline and remove the layer.

## Note

- For example or benchmark or script, to load package, use `import elastica as ea` and `import elastica_jax as eaj`
- Use `.venv` python.
- The goal is making good, intuitive, adaptable API to setup simulation. Currently focusing on integrating JAX on top of the existing framework.
- benchmark goes in @benchmark folder, not in backend.
- When you suggest cli with `uv run`, always add `--no-sync`.
- Code is being migrated from `elastica` to `elastica_jax`. Many existing modules are still in  `ea`, and JAX related codes are migrated here.
- I don't like these pattern: assume jax exist; else, let import raise be called, no need to handle separate exception.
```py
try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "MemoryBlockMuscleArmJax requires JAX. Install the optional GPU dependency."
    ) from exc
```
- Use `click` for CLI instead of `argparse`.
- Don't run large case yourself.

## Python coding style

> For repository-wide style convension, reference docs/style.md

This document provides guidelines for maintaining high-quality Python code. These rules MUST be followed by all AI coding agents and contributors.
All code you write must be readable and maintainable, and fully optimized.
"Readable and maintainable" includes:
- related codes are closer together
- tranversing multiple files into the folder is easy, but not to easy across folders. This means code in same folder-level or deeper is "detail" implementation, and code in higher folder-level or neighboring branch is "abstraction" in relative perspective.
    - Docstring and API should be written as if a person using the code from upper-folder needs to understand the code.

"Fully optimized" includes:
- algorithmic efficiency for memory and runtime
- using vectorized operation where appropriate
- following proper style conventions for the code language (e.g. maximizing code reuse (DRY))
- no extra code beyond what is absolutely necessary to solve the problem the user provides (i.e. no technical debt)
    - If a Python library can be imported to significantly reduce the amount of new code required to implement a function at optimal performance, and the library itself is small and does not have much overhead, ALWAYS use the library instead.
- Improve readability of the script by the following practice:
    - Simple functional implementation should be placed at the bottom.
    - Important class or method definiton comes first.

### Example / research run-script style

> If instruction is related to writing or organizing example case, take a look at [docs/agents-context/examples.md](docs/agents-context/examples.md).

### Before handing off to the user

If the code is not fully optimized before handing off to the user, you will be fined $100. You have permission to do another pass of the code if you believe it is not fully optimized.
- **MUST** use meaningful, descriptive variable and function names.
- **MUST** follow PEP 8 style guidelines
- **MUST** use 4 spaces for indentation (never tabs)
- **NEVER** use emoji, or unicode that emulates emoji (e.g. ✓, ✗). The only exception is when writing tests and testing the impact of multibyte characters.
- Use snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- Limit line length to 88 characters (ruff formatter standard)
- **MUST** avoid including redundant comments which are tautological or self-demonstating (e.g. cases where it is easily parsable what the code does at a glance so the comment does)
- **MUST** avoid including comments which leak what this file contains, or leak the original user prompt, ESPECIALLY if it's irrelevant to the output code.

### Preferred Tools

- Use `tqdm` to track long-running loops. For multiprocessing, make sure only single process is updating the progress bar. All progress bar should have disable capability, wired to verbose flag. The `description` of the progress bar should be contextually sensitive.
- Use `click` for CLI instead of `argparse`.

### Prefered export file types

- Use `mp4` for video files. Don't need to back-run to convert to gif, if ffmpeg is not installed. In that case, just let it raise error.

## Documentation

- **MUST** include docstrings for all public functions, classes, and methods
- **MUST** document function parameters, return values, and exceptions raised
- Keep comments up-to-date with code changes
- Include examples in docstrings for complex functions. Complex functions is when the behavior is not obvious at a glance.
- Prefer numpydoc style for docstrings.

## Type Hints

- **MUST** use type hints for all function signatures (parameters and return values)
- **AVOID** use `Any` type unless absolutely necessary
- Use `T | None` for nullable types

## Error Handling

- **NEVER** silently swallow exceptions without logging
- **MUST** never use bare `except:` clauses
- **MUST** catch specific exceptions rather than broad exception types
- **MUST** use context managers (`with` statements) for resource cleanup
- Provide meaningful error messages

## Function Design

- **MUST** keep functions focused on a single responsibility
- **NEVER** use mutable objects (lists, dicts) as default argument values
- Limit function parameters to 7 or fewer
- Return early to reduce nesting

## Class Design

- **MUST** keep classes focused on a single responsibility
- **MUST** keep `__init__` simple; avoid complex logic
- Use dataclasses for simple data containers
- Prefer composition over inheritance
- Avoid creating additional class functions if they are not necessary
- Use `@property` for computed attributes

## Testing

- **MUST** write unit tests for all new functions and classes
- **MUST** mock external dependencies (APIs, databases, file systems)
- **MUST** use pytest as the testing framework
- **NEVER** run tests you generate without first saving them as their own discrete file
- **NEVER** delete files created as a part of testing.
- Ensure the folder used for test outputs is present in `.gitignore`
- Follow the Arrange-Act-Assert pattern
- Do not commit commented-out tests

## Imports and Dependencies

- **MUST** avoid wildcard imports (`from module import *`)
- **MUST** document dependencies in `pyproject.toml`
- Organize imports: standard library, third-party, local imports

## Python Best Practices

- **NEVER** use mutable default arguments
- **MUST** use context managers (`with` statement) for file/resource management
- **MUST** use `is` for comparing with `None`, `True`, `False`
- **MUST** use f-strings for string formatting
- Use list comprehensions and generator expressions when appropriate.
- Use `enumerate()` instead of manual counter variables

## Benchmarking and Optimization

- **NEVER** run benchmarks in parallel, as the benchmarks will compete for resources and the results will be invalid
- **NEVER** game the benchmarks. Do not manipulate the benchmarks themselves to satisfy any required performance constraints
- If benchmarking against another crate or library, ensure the benchmarks are apples-to-apples comparisons
- Ensure benchmark tests are independent. If the tests are dependent due to a feature (e.g. caching), ensure the feature is disabled

## Security

- **NEVER** store secrets, API keys, or passwords in code. Only store them in `.env`
  - Ensure `.env` is declared in `.gitignore`.
  - **NEVER** print or log URLs to console if they contain an API key
- **MUST** use environment variables for sensitive configuration
- **NEVER** log sensitive information (passwords, tokens, PII)

## Version Control

- **MUST** write clear, descriptive commit messages
- **NEVER** commit commented-out code; delete it
- **NEVER** commit debug print statements or breakpoints
- **NEVER** commit credentials or sensitive data

## Tools

- **MUST** use Ruff for code formatting and linting (replaces Black, isort, flake8)
- **MUST** use mypy for static type checking
- Use `uv` for package management (faster alternative to pip)
- Use pytest for testing

## Before Committing

- [ ] All tests pass
- [ ] Type checking passes (mypy)
- [ ] Code formatter and linter pass (Ruff)
- [ ] All functions have docstrings and type hints
- [ ] No commented-out code or debug statements
- [ ] No hardcoded credentials

## Agent skills

### Issue tracker

Issues and specs live as markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.

## Version control

Use Graphite (`gt`) for the Git stacking workflow.
- Start a new stack from `main`; put each independent review item on its own
  branch and PR, with later branches based on the preceding branch.
- Stage only files belonging to the current review item. Never stage `.gitignore`.
- Use `gt create` to create stacked commits, `gt modify` for revisions, and
  `gt restack` to update downstream branches after changing a parent.
- Submit with `gt submit --stack`; preserve the PR base relationships and explain
  each branch's change and validation in its PR description.

---

**Remember:** Prioritize clarity and maintainability over cleverness. This is your core directive.
