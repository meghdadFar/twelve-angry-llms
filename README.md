# Twelve Angry LLMs

An extensible Python library for creating and using LLMs as judges. This library provides a framework for defining different types of judges, from those that return a simple score to those that provide a detailed, descriptive evaluation. It's designed to be modular, allowing you to easily connect or use different LLM providers like OpenAI, Hugging Face, or your own local models.

<!-- [![PyPI version](https://badge.fury.io/py/twelve-angry-llms.svg)](https://badge.fury.io/py/twelve-angry-llms) -->
<!-- [![Build Status](https://github.com/meghdadFar/twelve-angry-llms/actions/workflows/python-package.yml/badge.svg)](https://github.com/meghdadFar/twelve-angry-llms/actions/workflows/python-package.yml) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/twelve-angry-llms.svg)](https://pypi.org/project/twelve-angry-llms) -->

## Installation

```bash
pip install twelve-angry-llms
```

## Usage

Here's how to use the `DescriptiveJudge` with the `OpenAIProvider`:

```python
from twelve_angry_llms.providers.openai import OpenAIProvider
from twelve_angry_llms.judges.descriptive import DescriptiveJudge

# Make sure to set the OPENAI_API_KEY environment variable
openai_provider = OpenAIProvider()
descriptive_judge = DescriptiveJudge(llm_provider=openai_provider)

result = descriptive_judge.evaluate(
    generation="This is a test generation.",
    reference="This is a test reference.",
    context="This is a test context."
)

print(result.evaluation)
```

## Contributing

Contributions are welcome! Please see the [Contributing Guide](CONTRIBUTING.md) for more information.
