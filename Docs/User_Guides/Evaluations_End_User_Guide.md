# Evaluations User Guide - For End Users

## What Are Evaluations?

Evaluations help you measure the quality of AI-generated content like summaries, answers, and responses. Think of it as a grading system for AI - you can check if your AI is producing accurate, relevant, and well-written content.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Types of Evaluations](#types-of-evaluations)
3. [Step-by-Step Tutorials](#step-by-step-tutorials)
4. [Understanding Your Results](#understanding-your-results)
5. [Common Use Cases](#common-use-cases)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Cost Considerations](#cost-considerations)

---

## Getting Started

### What You Need
- Access to tldw_server (usually at `http://localhost:8000`)
- Your content to evaluate (text, summaries, Q&A pairs)
- (Optional) API key for AI providers like OpenAI or Anthropic

### Your First Evaluation in 3 Steps

**Step 1: Prepare Your Content**
You need two things:
- The original content (article, document, question)
- The AI-generated response (summary, answer)

**Step 2: Choose an Evaluation Type**
- **Summary Quality**: Use G-Eval for summaries
- **Q&A Quality**: Use RAG evaluation for question-answering
- **General Quality**: Use response quality evaluation

**Step 3: Run the Evaluation**
Send your content to the evaluation endpoint and get quality scores back.

---

## Types of Evaluations

### 1. G-Eval - For Summaries 📝

**What it does**: Grades summaries on four aspects:
- **Fluency** (1-3): Is it well-written?
- **Consistency** (1-5): Does it match the original facts?
- **Relevance** (1-5): Does it include important points?
- **Coherence** (1-5): Is it well-organized?

**When to use**: 
- Testing article summaries
- Evaluating meeting notes
- Checking document abstracts

**Example Score Interpretation**:
```
Fluency: 2.8/3 (Excellent grammar)
Consistency: 4.5/5 (Very accurate)
Relevance: 3.8/5 (Good, but missed some points)
Coherence: 4.2/5 (Well structured)
Overall: 82% - Good quality summary
```

### 2. RAG Evaluation - For Q&A Systems 🤖

**What it does**: Evaluates how well an AI answers questions using retrieved information.

**Metrics**:
- **Relevance**: Does the answer address the question?
- **Faithfulness**: Is it accurate to the source material?
- **Answer Similarity**: How close to the ideal answer?

**When to use**:
- Testing chatbots
- Evaluating search results
- Checking FAQ systems

### 3. Response Quality - For General Content ✍️

**What it does**: Checks if responses meet your specific requirements.

**Custom Criteria Examples**:
- Tone (professional, friendly, casual)
- Completeness (all points covered)
- Format (follows template)
- Accuracy (factually correct)

**When to use**:
- Email generation
- Content creation
- Customer service responses

---

## Step-by-Step Tutorials

### Tutorial 1: Evaluating an Article Summary

**Scenario**: You have an AI-generated summary of a news article and want to check its quality.

**Step 1: Prepare Your Data**
```
Original Article: "Climate change is affecting global weather patterns..."
AI Summary: "Climate change impacts weather worldwide..."
```

**Step 2: Send for Evaluation**

Using a simple web interface or tool:
```json
{
  "source_text": "Your full article text here...",
  "summary": "Your AI-generated summary here...",
  "api_name": "openai"
}
```

**Step 3: Interpret Results**
```
Results:
- Fluency: 85% (Well written)
- Consistency: 92% (Factually accurate)
- Relevance: 78% (Covered main points)
- Coherence: 88% (Good flow)

Overall Score: 86% - High Quality Summary ✓
```

### Tutorial 2: Testing Chatbot Responses

**Scenario**: Your chatbot answered a customer question, and you want to verify quality.

**Step 1: Collect the Information**
```
Question: "How do I reset my password?"
Context: [Your help documentation]
Bot Answer: "To reset your password, go to settings..."
```

**Step 2: Run RAG Evaluation**
```json
{
  "query": "How do I reset my password?",
  "retrieved_contexts": ["From help docs: Password reset steps..."],
  "generated_response": "To reset your password, go to settings..."
}
```

**Step 3: Review Scores**
```
Relevance: 95% - Directly answers the question
Faithfulness: 88% - Accurate to documentation
Overall: Excellent response quality
```

### Tutorial 3: Batch Evaluation for Multiple Items

**Scenario**: You have 10 summaries to evaluate at once.

**Step 1: Prepare Batch**
```json
{
  "evaluation_type": "geval",
  "items": [
    {"source_text": "Article 1", "summary": "Summary 1"},
    {"source_text": "Article 2", "summary": "Summary 2"},
    // ... more items
  ]
}
```

**Step 2: Run Batch Evaluation**
The system will evaluate all items in parallel.

**Step 3: Review Batch Results**
```
Batch Results:
- Total Items: 10
- Average Score: 84%
- Best: Item #3 (92%)
- Needs Improvement: Item #7 (68%)
```

---

## Understanding Your Results

### Score Interpretation Guide

**90-100%** 🌟 **Excellent**
- Production-ready quality
- Minimal or no improvements needed

**80-89%** ✅ **Good**
- High quality with minor issues
- Safe for most use cases

**70-79%** ⚠️ **Acceptable**
- Usable but needs improvement
- Review before critical use

**60-69%** ⚡ **Poor**
- Significant issues present
- Requires human review

**Below 60%** ❌ **Failing**
- Not suitable for use
- Major rework needed

### What Each Metric Means

**Fluency Issues** (Grammar/Writing):
- Spelling errors
- Grammar mistakes
- Awkward phrasing
- *Fix*: Use grammar correction tools

**Consistency Issues** (Accuracy):
- Factual errors
- Contradictions
- Hallucinations
- *Fix*: Verify facts, use better sources

**Relevance Issues** (Content):
- Missing key points
- Including unimportant details
- Off-topic content
- *Fix*: Adjust prompts, better context

**Coherence Issues** (Structure):
- Poor organization
- Jumbled ideas
- No logical flow
- *Fix*: Use structured templates

---

## Common Use Cases

### 1. Content Quality Assurance

**Use Case**: Ensure all AI-generated blog posts meet quality standards.

**Workflow**:
1. Generate content with AI
2. Run response quality evaluation
3. Only publish if score > 85%
4. Flag low scores for human review

### 2. A/B Testing Different Models

**Use Case**: Compare outputs from GPT-4 vs Claude.

**Process**:
1. Generate same content with both models
2. Evaluate both with same criteria
3. Compare scores
4. Choose better performing model

### 3. Continuous Improvement

**Use Case**: Track quality over time.

**Implementation**:
1. Evaluate samples daily/weekly
2. Track score trends
3. Identify problem areas
4. Adjust prompts/settings

### 4. Customer Service Quality

**Use Case**: Ensure chatbot responses are helpful.

**Checks**:
- Answer relevance (>90%)
- Tone appropriateness
- Completeness of response
- Policy compliance

---

## Best Practices

### 1. Choose the Right Evaluation Type

| Content Type | Best Evaluation | Key Metrics |
|-------------|-----------------|-------------|
| Summaries | G-Eval | Consistency, Relevance |
| Q&A | RAG | Faithfulness, Relevance |
| Creative Writing | Response Quality | Custom criteria |
| Technical Docs | Response Quality | Accuracy, Completeness |
| Chat Responses | RAG or Response | Relevance, Tone |

### 2. Set Appropriate Thresholds

- **Critical Content** (legal, medical): >90% required
- **Customer-Facing** (support, sales): >80% recommended
- **Internal Use** (notes, drafts): >70% acceptable
- **Experimental** (testing): Any score for learning

### 3. Sample Size Guidelines

- **Quick Check**: 5-10 samples
- **Confidence**: 20-50 samples
- **Statistical Significance**: 100+ samples

### 4. Cost Optimization

- **Development**: Use cheaper models (GPT-3.5)
- **Production**: Use better models (GPT-4)
- **Batch Processing**: Evaluate multiple items together
- **Caching**: Store results to avoid re-evaluation

---

## Troubleshooting

### Common Issues and Solutions

#### "Rate limit exceeded"
**Problem**: Too many requests too quickly.
**Solution**: 
- Wait 60 seconds before retrying
- Use batch evaluation for multiple items
- Spread requests over time

#### "Low scores across all evaluations"
**Problem**: Systematic quality issues.
**Check**:
- Is the source content clear?
- Are you using appropriate prompts?
- Is the model suitable for this task?

#### "Inconsistent scores for same content"
**Problem**: Evaluation variability.
**Solution**:
- Run multiple evaluations and average
- Use temperature=0 for consistency
- Check for time-of-day effects

#### "Evaluation takes too long"
**Problem**: Performance issues.
**Try**:
- Reduce text length
- Use batch processing
- Check server resources

---

## Cost Considerations

### Typical Costs per Evaluation

| Provider | Per Evaluation | 1000 Evaluations |
|----------|---------------|------------------|
| GPT-3.5 | $0.001-0.003 | $1-3 |
| GPT-4 | $0.01-0.03 | $10-30 |
| Claude | $0.008-0.024 | $8-24 |
| Local LLM | Free | Free |

### Cost Optimization Tips

1. **Use Sampling**: Don't evaluate everything
   - Evaluate 10% of production content
   - Focus on high-risk content

2. **Choose Models Wisely**:
   - GPT-3.5 for development/testing
   - GPT-4 for production/critical
   - Local models for high volume

3. **Batch Processing**: 
   - Group evaluations together
   - Run during off-peak hours

4. **Smart Caching**:
   - Store evaluation results
   - Re-use for similar content

---

## Quick Reference

### API Endpoints

| Task | Endpoint | Method |
|------|----------|--------|
| Evaluate Summary | `/evaluations/geval` | POST |
| Evaluate Q&A | `/evaluations/rag` | POST |
| Check Quality | `/evaluations/response-quality` | POST |
| Batch Evaluate | `/evaluations/batch` | POST |
| View History | `/evaluations/history` | POST |

### Required Fields

**For Summaries**:
- `source_text`: Original document
- `summary`: AI-generated summary

**For Q&A**:
- `query`: The question
- `retrieved_contexts`: Source information
- `generated_response`: AI answer

**For General**:
- `prompt`: What was requested
- `response`: What AI produced

---

## Examples for Different Use Cases

### Example 1: News Summary Evaluation
```python
# Evaluating a news article summary
evaluation = {
    "source_text": "Full article about economic trends...",
    "summary": "The economy shows mixed signals with...",
    "metrics": ["consistency", "relevance"]
}
# Expected: High consistency (facts), high relevance (key points)
```

### Example 2: Customer Support Response
```python
# Evaluating a support chatbot response
evaluation = {
    "query": "My order hasn't arrived",
    "retrieved_contexts": ["Shipping policy: Orders arrive in 3-5 days..."],
    "generated_response": "I understand your concern. Orders typically..."
}
# Expected: High relevance, appropriate tone
```

### Example 3: Technical Documentation
```python
# Evaluating technical writing
evaluation = {
    "prompt": "Explain how to install Python",
    "response": "To install Python, follow these steps...",
    "evaluation_criteria": {
        "completeness": "All steps included",
        "accuracy": "Commands are correct",
        "clarity": "Easy to follow"
    }
}
# Expected: High accuracy, complete instructions
```

---

## Getting Help

### Resources
- **API Documentation**: See [API Reference](../API-related/Evaluations_API_Reference.md)
- **Technical Setup**: See [Developer Guide](../Code_Documentation/Evaluations_Developer_Guide.md)
- **Deployment**: See [Production Guide](./Evaluations_Production_Deployment_Guide.md)

### Support Checklist
Before requesting help:
1. Check your evaluation scores and explanations
2. Verify your API key is correct
3. Ensure content is within size limits
4. Try with a simple example first
5. Check the health endpoint: `/health/evaluations`

### Contact
- GitHub Issues: Report bugs or request features
- Documentation: Check `/Docs/` for detailed guides
- Logs: Include evaluation_id in support requests

---

## Summary

Evaluations help ensure your AI-generated content meets quality standards. Start with simple evaluations, understand the scores, and gradually implement more sophisticated quality checks. Remember:

- 🎯 Choose the right evaluation type for your content
- 📊 Set appropriate quality thresholds
- 💰 Optimize costs with smart sampling
- 📈 Track quality trends over time
- 🔧 Adjust based on evaluation feedback

With regular evaluation, you can maintain high-quality AI outputs and build trust in your automated systems.