# Evaluation Plans

----------------------------------------------------------------------------------------------------------------
## Table of Contents
- []()
- []()
- []()
- []()
- []()
----------------------------------------------------------------------------------------------------------------


Benchmarking with distilabel
    https://distilabel.argilla.io/latest/sections/pipeline_samples/examples/benchmarking_with_distilabel/


Chat arena
    https://github.com/lm-sys/FastChat
LLM-as-judge
    https://huggingface.co/learn/cookbook/en/llm_judge



----------------------------------------------------------------------------------------------------------------
### Introduction

----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### Evaluation Plan
- Model Evaluation
- Search Evaluation (Assuming test data set)
  - Retrieval Evaluation First
    - Identify keyword search success rate
    - Identify Semantic search success rate
    - Identify Hybrid search success rate
- RAG evaluation
  - Generation Eval Second
    - Create synthetic eval data by taking a document and generating questions about the document using an LLM.
  - User queries + ranking = Eval data
  - Integrate 'real' eval data into synthetic eval data
- Embeddings Retrieval Evaluation
- LLM Evaluations
- Evaluation Metrics
- Things to look out for
  - https://arxiv.org/abs/2411.03923
- **Reproducability**
  - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/troubleshooting/troubleshooting-reproducibility.md
- **Designing Evaluations**
  - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/model-as-a-judge/designing-your-evaluation-prompt.md
  - https://eugeneyan.com/assets/llm-eval-tree.jpg
- Human Evaluation
  - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/human-evaluation/using-human-annotators.md
----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### Model Evaluation
- **Metrics**
  1. Answer Relevancy
     * Does the LLM give a relevant answer to the question?
  2. Correctness
     - Is the LLM output correct regarding a 'ground truth'
  3. Confabulation-Rate
     - How often does the LLM make up information?
  4. Contextual Relevancy
     - How relevant is the returned content to the context of the question?
  5. Bias
  6. Task-Specific

- **Metrics should be:**
  1. Quantifiable
  2. Reproducible
  3. Sensitive
  4. Specific
  5. Interpretable

- **Evaluation Methodologies**
    - G-Eval

- **Frameworks**
    - OpenCompass
    - DeepEval
    - lm-eval-harness
    - https://github.com/EleutherAI/lm-evaluation-harness
    - https://github.com/nexusflowai/NexusBench
- Pop Culture
- 'Reasoning'
    - MMLU-Pro
        - https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        - https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main
- **Role Play**
  - Discussion from different PoV Facilitation
    - https://github.com/Neph0s/awesome-llm-role-playing-with-persona?tab=readme-ov-file
    - https://github.com/lawraa/LLM-Discussion
    - https://github.com/InteractiveNLP-Team/RoleLLM-public
  - Role Adherence
    - StickToYourRole
      - https://huggingface.co/datasets/flowers-team/StickToYourRole
      - https://huggingface.co/datasets/flowers-team/StickToYourRole
      - https://arxiv.org/abs/2402.14846
      - https://flowers-team-sticktoyourroleleaderboard.hf.space/about
    - PingPong Bench
      - https://github.com/IlyaGusev/ping_pong_bench
      - https://ilyagusev.github.io/ping_pong_bench/
    - DeepEval
      - https://docs.confident-ai.com/docs/metrics-role-adherence
    - General Research / Unsorted
      - https://arxiv.org/html/2406.00627v1
      - https://mp.weixin.qq.com/s/H2KNDGRNHktHiQc3sayFsA
      - https://mp.weixin.qq.com/s/2lbCMo64-nU5yRz1cLQxYA
      - https://mp.weixin.qq.com/s/E5qp5YPYPVaLM07OumDTRw
      - https://mp.weixin.qq.com/s/yoM-srJYGGfyd1VXirg_Hg
    - RP-Bench
      - https://boson.ai/rpbench-blog/
  - Collections of research
    - https://github.com/MiuLab/PersonaLLM-Survey
  - Notes
    - https://ianbicking.org/blog/2024/04/roleplaying-by-llm
- **Links**
  - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/tips-and-tricks.md


----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### Search Evaluation

- **Basic Search Eval via swyx**
  1. Take your existing text chunks
  2. Generate questions that could be answered by each chunk
  3. Store {question, chunk_id} pairs
  4. Test: Does your retrieval system find the source chunk?

----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### RAG Evaluation
https://arxiv.org/abs/2411.03538
https://archive.is/OtPVh
https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/
https://github.com/TonicAI/tonic_validate
https://github.com/D-Star-AI/KITE

- **Metrics**
    1. Answer Consistency
       * Whether there is information in the LLM answer that does not come from the context.
    2. Answer relevancy
    3. Answer Similarity Score
       * How well the reference answer matches the LLM answer.
    4. Retrieval Precision
       * Whether the context retrieved is relevant to answer the given question.
    5. Augmentation precision
       * Whether the relevant context is in the LLM answer.
    6. Augmentation Accuracy
       * Whether all the context is in the LLM answer.
    7. Contextual Recall
    8. Latency
       * How long it takes for the LLM to complete a request.
    9. Contains Text
         * Whether the LLM answer contains the specific text.



RAG Eval Plan:
    The generic idea however: you take a (full, unchunked) document and ask an LLM to generate a question with that document as well as give the factual answer to it. Enforce via prompts to make it use the document only and make it as hard as you want (eg. maybe sometimes you want it to consider 2 documents and make a question that uses bits of both). This gives you a ground truth dataset.
    You then kick off your RAG pipeline on your documents. They will be chunked, indexed and stored. Then you fire all the questions of your ground truth set at your RAG pipeline and check 1. If it found chunks of the correct document and 2. Ask an LLM various evaluation questions about the generated answer vs. the ground truth answer ( like: how related are they, is there content in the answer that is not in the doc chunks, etc).
    This gives you a good idea how well your retrieval (and with that, indexing) works, and how well your full pipeline works. As a bonus you could also keep track of which chunk(s) the ground truth answer was based on and use that for retrieval evaluation too. 


- **When to move on from Primarily Synthetic Data to Real User Data**
    - 80% recall on synthetic tests
    - Good understanding of failure cases
    - Clear chunking strategy
    - Growing set of real user examples

----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### Embeddings Retrieval Evaluation


----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### LLM Evaluations


----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### VLM Evaluations
https://arxiv.org/abs/2411.04075
https://arxiv.org/abs/2411.02571


- xkcd bench: https://github.com/arnokha/explain-xkcd-with-llms
- 
----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### Task Evaluations

- Long Document Comprehension
  - https://arxiv.org/html/2411.01106v1
- Math Eval
  - https://arxiv.org/abs/2411.04872



- https://arxiv.org/abs/2411.02305

----------------------------------------------------------------------------------------------------------------

LLM-as-a-Judge
  https://hamel.dev/blog/posts/llm-judge
Quant Eval
https://arxiv.org/abs/2411.02355


Finetuning
    https://huggingface.co/learn/cookbook/enterprise_cookbook_argilla
    https://aclanthology.org/2024.cl-3.1/
    https://scale.com/guides/data-labeling-annotation-guide
    https://aclanthology.org/2024.naacl-long.126/
    https://distilabel.argilla.io/latest/
    https://distilabel.argilla.io/latest/sections/pipeline_samples/papers/ultrafeedback/
