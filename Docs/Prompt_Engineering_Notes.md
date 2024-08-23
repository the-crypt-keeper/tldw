# Prompt Engineering Notes

## Table of Contents
- [101](#101)
- [Prompting Techniques](#prompt-techniques)
- [Prompt Samples and Random notes](#prompt-samples)







--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 101<a name="101"></a>
- **What is Prompt Engineering?**
  - Prompt engineering is the process of designing and optimizing prompts to guide language models (LLMs) in generating desired outputs. It involves crafting instructions, examples, and context to elicit specific responses from the model.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Prompting Techniques<a name="prompt-techniques"></a>

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Prompt Samples and Random notes<a name="prompt-samples"></a>

I don't remember where I got this from....
```
1. Focus on Prompting Techniques:

   1.1. Start with n-shot prompts to provide examples demonstrating tasks.
   1.2. Use Chain-of-Thought (CoT) prompting for complex tasks, making instructions specific.
   1.3. Incorporate relevant resources via Retrieval Augmented Generation (RAG).

2. Structure Inputs and Outputs:

   2.1. Format inputs using serialization methods like XML, JSON, or Markdown.
   2.2. Ensure outputs are structured to integrate seamlessly with downstream systems.

3. Simplify Prompts:

   3.1. Break down complex prompts into smaller, focused ones.
   3.2. Iterate and evaluate each prompt individually for better performance.

4. Optimize Context Tokens:

   4.1. Minimize redundant or irrelevant context in prompts.
   4.2. Structure the context clearly to emphasize relationships between parts.

5. Leverage Information Retrieval/RAG:

   5.1. Use RAG to provide the LLM with knowledge to improve output.
   5.2. Ensure retrieved documents are relevant, dense, and detailed.
   5.3. Utilize hybrid search methods combining keyword and embedding-based retrieval.

6. Workflow Optimization:

   6.1. Decompose tasks into multi-step workflows for better accuracy.
   6.2. Prioritize deterministic execution for reliability and predictability.
   6.3. Use caching to save costs and reduce latency.

7. Evaluation and Monitoring:

   7.1. Create assertion-based unit tests using real input/output samples.
   7.2. Use LLM-as-Judge for pairwise comparisons to evaluate outputs.
   7.3. Regularly review LLM inputs and outputs for new patterns or issues.

8. Address Hallucinations and Guardrails:

   8.1. Combine prompt engineering with factual inconsistency guardrails.
   8.2. Use content moderation APIs and PII detection packages to filter outputs.

9. Operational Practices:

   9.1. Regularly check for development-prod data skew.
   9.2. Ensure data logging and review input/output samples daily.
   9.3. Pin specific model versions to maintain consistency and avoid unexpected changes.

10. Team and Roles:

    10.1. Educate and empower all team members to use AI technology.
    10.2. Include designers early in the process to improve user experience and reframe user needs.
    10.3. Ensure the right progression of roles and hire based on the specific phase of the project.

11. Risk Management:

    11.1. Calibrate risk tolerance based on the use case and audience.
    11.2. Focus on internal applications first to manage risk and gain confidence before expanding to customer-facing use cases.
```