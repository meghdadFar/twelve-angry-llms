# Twelve Angry LLMs

An extensible Python library for using LLMs as judges.

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
