# Evaluations Module - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Available Benchmarks](#available-benchmarks)
4. [Running Evaluations](#running-evaluations)
5. [Creating Custom QA Benchmarks](#creating-custom-qa-benchmarks)
6. [Using WordBench](#using-wordbench)
7. [Understanding Results](#understanding-results)
8. [CLI Reference](#cli-reference)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Introduction

The Evaluations module provides a comprehensive system for evaluating language models across various benchmarks and metrics. Whether you want to test your model on established benchmarks like MMLU Pro, create your own QA tests, or analyze token prediction behavior, this module has you covered.

### Key Features
- **Pre-configured Benchmarks**: MMLU Pro, GPQA, Simple Bench, and more
- **Custom Evaluations**: Create your own QA benchmarks easily
- **Token Analysis**: WordBench for analyzing next-token predictions
- **Multiple Scoring Methods**: Exact match, similarity, LLM judge
- **Comprehensive Analysis**: Detailed metrics and weakness identification
- **CLI and API Access**: Use via command line or programmatically

## Getting Started

### Installation

The Evaluations module is included with tldw_server. Ensure you have the CLI tools installed:

```bash
# Install with evaluations support
pip install -e ".[evals]"

# Verify installation
tldw-evals --help
```

### Quick Start

1. **Check system health:**
```bash
tldw-evals health
```

2. **List available benchmarks:**
```bash
tldw-evals list-benchmarks --detailed
```

3. **Run your first evaluation:**
```bash
tldw-evals run simple_bench --limit 10
```

## Available Benchmarks

### Pre-configured Benchmarks

| Benchmark | Type | Description | Size |
|-----------|------|-------------|------|
| **MMLU Pro** | Multiple Choice | Enhanced MMLU with 10 options and reasoning | 12,000 questions |
| **GPQA Diamond** | Multiple Choice | Graduate-level science questions | Variable |
| **Simple Bench** | Multiple Choice | Reasoning problems with scenarios | ~100 questions |
| **BFCL** | Function Calling | Berkeley Function Calling tests | Variable |
| **SWE-bench** | Code Generation | Software engineering tasks | Variable |
| **MASK** | Honesty | Honesty and consistency evaluation | Variable |
| **WordBench** | Token Analysis | Next token prediction analysis | Custom prompts |

### Benchmark Types

1. **Multiple Choice**: Questions with predefined answer options
2. **Code Generation**: Programming tasks with test cases
3. **Function Calling**: API/function call generation
4. **Instruction Following**: Constraint satisfaction tests
5. **Token Analysis**: Probability distribution analysis

## Running Evaluations

### Basic Usage

```bash
# Run a benchmark with default settings
tldw-evals run mmlu_pro

# Limit the number of samples
tldw-evals run mmlu_pro --limit 100

# Save results to file
tldw-evals run mmlu_pro --output results.json

# Use a specific API/model
tldw-evals run simple_bench --api anthropic --api-key YOUR_KEY
```

### Advanced Options

```bash
# Resume from previous run
tldw-evals run mmlu_pro --resume-from partial_results.json

# Run with parallel processing
tldw-evals run simple_bench --parallel 8

# Validate benchmark before running
tldw-evals validate mmlu_pro --samples 5
```

### Using Different APIs

The module supports multiple LLM providers:

```bash
# OpenAI (default)
tldw-evals run benchmark_name --api openai

# Anthropic
tldw-evals run benchmark_name --api anthropic

# Local models
tldw-evals run benchmark_name --api local-llm

# Custom OpenAI-compatible endpoint
tldw-evals run benchmark_name --api custom --endpoint https://your-api.com
```

## Creating Custom QA Benchmarks

### Method 1: Using the Template

1. **Copy the template:**
```bash
cp configs/simple_qa_template.yaml configs/my_benchmark.yaml
```

2. **Edit the configuration:**
```yaml
name: my_benchmark
description: My custom QA benchmark
dataset_source: ./data/my_questions.json
evaluation_params:
  scoring_method: llm_judge  # or exact_match, contains, similarity
```

3. **Create your questions file:**
```json
{
  "questions": [
    {
      "question": "What is your question?",
      "answer": "The expected answer",
      "category": "topic",
      "difficulty": "easy",
      "alternative_answers": ["Other valid answer"]
    }
  ]
}
```

4. **Run your benchmark:**
```bash
tldw-evals run my_benchmark
```

### Method 2: Using Python Helper

```python
from qa_benchmark_helper import QADatasetBuilder

# Create from Q&A pairs
qa_pairs = [
    ("Question 1?", "Answer 1"),
    ("Question 2?", "Answer 2")
]

dataset = QADatasetBuilder.create_from_pairs(qa_pairs)
QADatasetBuilder.save_dataset(dataset, "my_benchmark.json")
```

### Method 3: Quick CLI Creation

```bash
# Register a custom benchmark
tldw-evals register my_benchmark \
  --config configs/my_benchmark.yaml \
  --source data/questions.json \
  --type qa_evaluation
```

## Using WordBench

WordBench analyzes next-token predictions and their probability distributions.

### Running WordBench

```bash
# Use example prompts
tldw-evals run wordbench

# Use custom prompts file
tldw-evals run wordbench --prompts my_prompts.txt

# Analyze specific prompts
echo "The sky is" | tldw-evals run wordbench --stdin
```

### Understanding WordBench Output

```
Prompt: 'The sky is'
Generated: 'blue'

Top 10 Next Token Predictions:
----------------------------------------
 1. 'blue': 45.23% (prob: 0.4523)
 2. 'clear': 12.34% (prob: 0.1234)
 3. 'dark': 8.91% (prob: 0.0891)
 ...

Distribution Analysis:
----------------------------------------
Entropy: 2.145              # Uncertainty measure
Top token probability: 0.452 # Confidence
Concentration: medium        # How focused the distribution is
```

### WordBench Use Cases

- **Bias Detection**: Analyze token preferences in different contexts
- **Model Comparison**: Compare probability distributions across models
- **Uncertainty Analysis**: Identify where models are most/least confident
- **Behavioral Study**: Understand model decision-making patterns

## Understanding Results

### Metrics Explained

| Metric | Description | Good Range |
|--------|-------------|------------|
| **Accuracy** | Percentage of correct answers | > 80% |
| **Average Score** | Mean score across all questions | > 0.75 |
| **Entropy** | Uncertainty in predictions (WordBench) | Context-dependent |
| **F1 Score** | Balance of precision and recall | > 0.8 |
| **Perfect Scores** | Questions answered perfectly | Higher is better |

### Results File Structure

```json
{
  "benchmark": "benchmark_name",
  "num_samples": 100,
  "results": [
    {
      "question_id": 1,
      "score": 0.85,
      "model_answer": "...",
      "expected_answer": "...",
      "explanation": "..."
    }
  ],
  "summary": {
    "average_score": 0.82,
    "by_category": {...},
    "by_difficulty": {...}
  }
}
```

### Analyzing Results

```bash
# Generate detailed report
tldw-evals analyze results.json --format detailed

# Compare multiple runs
tldw-evals compare run1.json run2.json run3.json

# Export to CSV
tldw-evals export results.json --format csv --output results.csv
```

## CLI Reference

### Main Commands

| Command | Description | Example |
|---------|-------------|---------|
| `list-benchmarks` | List available benchmarks | `tldw-evals list-benchmarks -d` |
| `run` | Run a benchmark | `tldw-evals run mmlu_pro` |
| `validate` | Validate benchmark configuration | `tldw-evals validate my_benchmark` |
| `register` | Register custom benchmark | `tldw-evals register name --config file.yaml` |
| `health` | Check system health | `tldw-evals health` |

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Config file path | Auto-detect |
| `--log-level` | Logging level | INFO |
| `--quiet` | Suppress output | False |
| `--output` | Output file path | None |

### Run Command Options

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--limit` | `-l` | Limit samples | `--limit 100` |
| `--api` | `-a` | API to use | `--api openai` |
| `--api-key` | | API key | `--api-key sk-...` |
| `--parallel` | `-p` | Parallel workers | `--parallel 4` |
| `--resume-from` | | Resume file | `--resume-from partial.json` |
| `--save-progress` | | Save after each batch | `--save-progress` |

## Examples

### Example 1: Testing Basic Knowledge

```bash
# Create a simple knowledge test
cat > knowledge_test.json << 'EOF'
{
  "questions": [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "Largest planet?", "answer": "Jupiter"}
  ]
}
EOF

# Run the test
tldw-evals run qa_evaluation --source knowledge_test.json
```

### Example 2: Category-Based Analysis

```python
from qa_benchmark_helper import QABenchmarkAnalyzer

# Load results
with open('results.json') as f:
    results = json.load(f)['results']

# Analyze by category
metrics = QABenchmarkAnalyzer.calculate_metrics(results)
print(f"Math performance: {metrics['by_category']['math']['average']:.1%}")
print(f"Science performance: {metrics['by_category']['science']['average']:.1%}")
```

### Example 3: Comparing Models

```bash
# Run same benchmark on different models
tldw-evals run simple_bench --api openai --output openai_results.json
tldw-evals run simple_bench --api anthropic --output anthropic_results.json

# Compare results
tldw-evals compare openai_results.json anthropic_results.json
```

### Example 4: Custom Scoring

```yaml
# Custom scoring configuration
evaluation_params:
  scoring_method: llm_judge
  judge_prompt_template: |
    Evaluate this answer for accuracy and completeness.
    Question: {question}
    Expected: {expected_answer}
    Model: {model_answer}
    
    Rate 0-1 based on:
    - Factual accuracy (50%)
    - Completeness (30%)
    - Clarity (20%)
    
    SCORE: X.X EXPLANATION: ...
```

## Troubleshooting

### Common Issues

**Issue: "API key not found"**
```bash
# Set API key via environment
export OPENAI_API_KEY=your_key

# Or pass directly
tldw-evals run benchmark --api-key your_key
```

**Issue: "Dataset not loading"**
```bash
# Validate dataset format
tldw-evals validate my_benchmark --samples 3

# Check file path
ls -la data/my_questions.json
```

**Issue: "Low scores on benchmark"**
- Check if prompt template matches model's expected format
- Verify API parameters (temperature, max_tokens)
- Review scoring method appropriateness
- Examine failed examples for patterns

**Issue: "Out of memory"**
```bash
# Reduce batch size
tldw-evals run benchmark --parallel 1 --limit 50

# Process in chunks
tldw-evals run benchmark --limit 100 --save-progress
```

### Getting Help

1. **Check logs:**
```bash
tldw-evals run benchmark --log-level DEBUG
```

2. **Validate configuration:**
```bash
tldw-evals validate benchmark_name
```

3. **Test with small sample:**
```bash
tldw-evals run benchmark --limit 5 --log-level DEBUG
```

4. **Report issues:**
- GitHub: https://github.com/your-repo/issues
- Include: Error message, config file, sample data

## Best Practices

### Creating Good Benchmarks

1. **Clear Questions**: Avoid ambiguity
2. **Complete Answers**: Include all acceptable variations
3. **Proper Categories**: Group related questions
4. **Balanced Difficulty**: Mix easy, medium, hard
5. **Regular Updates**: Keep questions current

### Running Evaluations

1. **Start Small**: Test with limited samples first
2. **Save Progress**: Use `--save-progress` for long runs
3. **Monitor Resources**: Watch memory and API usage
4. **Version Control**: Track benchmark and result files
5. **Document Changes**: Note configuration modifications

### Analyzing Results

1. **Look for Patterns**: Identify systematic weaknesses
2. **Compare Baselines**: Track improvement over time
3. **Statistical Significance**: Use adequate sample sizes
4. **Category Analysis**: Understand domain-specific performance
5. **Error Analysis**: Examine failed cases closely

## Next Steps

- Explore advanced benchmarks in the [Developer Guide](EVALS_DEVELOPER_GUIDE.md)
- Create domain-specific evaluations for your use case
- Contribute new benchmarks to the community
- Join discussions on benchmark design and methodology