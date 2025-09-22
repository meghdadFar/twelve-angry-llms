# Contributing to Twelve Angry LLMs

First off, thank you for considering contributing to Twelve Angry LLMs! It's people like you that make such an open-source project possible.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

    ```bash
    git clone https://github.com/your_username/twelve-angry-llms.git
    ```

3. **Set up a virtual environment** and install the dependencies. We recommend using `uv`:

    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

## Adding a New Judge

To add a new judge, you should:

1. Create a new file in `src/twelve_angry_llms/judges/`.
2. Create a new `EvaluationResult` model in `src/twelve_angry_llms/results/` that defines the structure of your judge's output.
3. Create a new class that inherits from `Judge` and implements the `evaluate` method. The `evaluate` method should return an instance of your `EvaluationResult` model.

## Adding a New LLM Provider

To add a new LLM provider, you should:

1. Create a new file in `src/twelve_angry_llms/providers/`.
2. Create a new class that inherits from `LLMProvider` and implements the `get_response` method.

## Code Style

We use the **Google Python Style Guide** for docstrings. Please make sure your contributions adhere to this style.

We use `ruff` for linting and formatting. Before submitting a pull request, please run:

```bash
ruff check .
ruff format .
```

## Submitting a Pull Request

1. Create a new branch for your feature: `git checkout -b my-new-feature`
2. Commit your changes: `git commit -am 'Add some feature'`
3. Push to the branch: `git push origin my-new-feature`
4. Submit a pull request.

We'll review your pull request as soon as possible. Thank you for your contribution!
