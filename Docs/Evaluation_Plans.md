# Evaluation Plans

----------------------------------------------------------------------------------------------------------------
## Table of Contents
- [Introduction](#introduction)
- [Evaluation Plan](#evaluation-plan)
- [Model Evaluation](#model-evaluation)
- [Search Evaluations](#search-eval)
- [RAG Evaluation](#rag-eval)
- [Embeddings Retrieval Evaluation](#embeddings-retrieval-eval)
- [VLM Evaluations](#vlm-evals)
----------------------------------------------------------------------------------------------------------------


Benchmarking with distilabel
    https://distilabel.argilla.io/latest/sections/pipeline_samples/examples/benchmarking_with_distilabel/


Chat arena
    https://github.com/lm-sys/FastChat
LLM-as-judge
    https://huggingface.co/learn/cookbook/en/llm_judge
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


----------------------------------------------------------------------------------------------------------------
### <a name="introduction"></a> Introduction

----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="evaluation-plan"></a> Evaluation Plan
- Need to review this: https://github.com/tianyi-lab/BenTo (Why isn't it being used???)
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
### <a name="model-evaluation"></a> Model Evaluation
- https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
- **101**
    - https://github.com/huggingface/evaluation-guidebook
- **Metrics**
    1. Answer Relevancy
         * Does the LLM give a relevant answer to the question?
         * DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/answer_relevancy
    2. Correctness
         - Is the LLM output correct regarding a 'ground truth'
    3. Confabulation-Rate
         - How often does the LLM make up information?
   4. Contextual Relevancy
         - How relevant is the returned content to the context of the question?
         * DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/contextual_relevancy
   5. Bias
         - DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/bias
   6. Task-Specific
  7. Conversation Intent
     * DeepEval https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/conversation_relevancy

- **Metrics should be:**
    1. Quantifiable
    2. Reproducible
    3. Sensitive
    4. Specific
    5. Interpretable

- **Evaluation Methodologies**
    - G-Eval
      - https://ai.plainenglish.io/how-to-evaluate-fluency-in-llms-and-why-g-eval-doesnt-work-3635298e9a43?gi=463011e4689d
    - QAG - QA Generation
          - https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task

- **Frameworks**
    - OpenCompass
    - DeepEval
    - lm-eval-harness
        - https://github.com/EleutherAI/lm-evaluation-harness
    - lighteval
        - https://github.com/huggingface/lighteval
    - NexusBench
        - https://github.com/nexusflowai/NexusBench
    - Athina.ai
        - https://docs.athina.ai/evals/preset-evals/overview
    - SimpleEval
        - https://github.com/openai/simple-evals/tree/main
    - EvalAI
    - FlashRAG
        - https://github.com/RUC-NLPIR/FlashRAG

- **Citations**
    - L-CiteEval
        - https://huggingface.co/papers/2410.02115
        - https://github.com/ZetangForward/L-CITEEVAL
- **Coding Ability**
    - Aider
        - https://github.com/Aider-AI/aider/tree/main/benchmark
    - CodeMMLU
        - https://arxiv.org/abs/2410.01999
        - https://github.com/FSoft-AI4Code/CodeMMLU
    - StackUnseen
        - https://prollm.toqan.ai/leaderboard/stack-unseen
- **Confabulation-Rate**
    - https://arxiv.org/abs/2409.11353
    - https://github.com/sylinrl/TruthfulQA
- **Context**
    - RULER
    - InfiniteBench
        - https://arxiv.org/abs/2402.13718
        - https://github.com/OpenBMB/InfiniteBench
        - `Welcome to InfiniteBench, a cutting-edge benchmark tailored for evaluating the capabilities of language models to process, understand, and reason over super long contexts (100k+ tokens). Long contexts are crucial for enhancing applications with LLMs and achieving high-level interaction. InfiniteBench is designed to push the boundaries of language models by testing them against a context length of 100k+, which is 10 times longer than traditional datasets.`
    - Babilong
        * `BABILong is a novel generative benchmark for evaluating the performance of NLP models in processing arbitrarily long documents with distributed facts.`
        * `BABILong consists of 20 tasks designed for evaluation of basic aspects of reasoning. The bAbI tasks are generated by simulating a set of characters and objects engaged in various movements and interactions with each other in multiple locations. Each interaction is represented by a fact, e.g. ”Mary travelled to the office”, and the task is to answer a question using the facts from the current simulation, for instance, ”Where is Mary?”. The bAbI tasks vary based on the number of facts, question complexity and the aspects of reasoning.`
        * https://huggingface.co/datasets/RMT-team/babilong
        * https://github.com/booydar/babilong
    - LongICLBench
        * `We created LongICLBench to conduct comprehensive evaluations of Large Language Models (LLMs) on extreme-label classification challenges with in-context learning. We compiled six datasets that encompass a broad spectrum of labels, ranging from 28 to 174 categories, and varied the lengths of input (from few-shot demonstrations) between 2K and 50K tokens to ensure thorough testing`
        * https://github.com/TIGER-AI-Lab/LongICLBench
        * https://arxiv.org/abs/2404.02060 - `Large Language Models (LLMs) have made significant strides in handling long sequences. Some models like Gemini could even to be capable of dealing with millions of tokens. However, their performance evaluation has largely been confined to metrics like perplexity and synthetic tasks, which may not fully capture their true abilities in more challenging, real-world scenarios. We introduce a benchmark (LongICLBench) for long in-context learning in extreme-label classification using six datasets with 28 to 174 classes and input lengths from 2K to 50K tokens. Our benchmark requires LLMs to comprehend the entire input to recognize the massive label spaces to make correct predictions. We evaluate on 15 long-context LLMs and find that they perform well on less challenging classification tasks with smaller label space and shorter demonstrations. However, they struggle with more challenging task like Discovery with 174 labels, suggesting a gap in their ability to process long, context-rich sequences. Further analysis reveals a bias towards labels presented later in the sequence and a need for improved reasoning over multiple pieces of information. Our study reveals that long context understanding and reasoning is still a challenging task for the existing LLMs. We believe LongICLBench could serve as a more realistic evaluation for the future long-context LLMs.`
    - Snorkel Working Memory Test
        * https://github.com/snorkel-ai/long-context-eval
        * https://arxiv.org/pdf/2407.03651
        * `This repository provides a Snorkel Working Memory Test (SWiM) to evaluate the long context capabilities of large language models (LLMs) on your own data and tasks. This is an improvement to the "needle in a haystack" (NIAH) test, where the haystack is your own set of documents, and the needles are one or more answer (complete) documents based on which the question is posed.`
    - HelloBench
        - https://github.com/Quehry/HelloBench
        - https://arxiv.org/abs/2409.16191
- **Creative Writing**
    - EQ Bench
        - https://eqbench.com/creative_writing.html
- **Culture**
    - https://arxiv.org/html/2305.14328v2
    - https://arxiv.org/abs/2411.06032
    - https://arxiv.org/abs/2410.02677
    - User-Centric Evaluation of LLMs
        - https://github.com/Alice1998/URS
    - https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks
- **Deceptiveness/Gullibility**
    - https://github.com/lechmazur/deception
- **Math Eval**
    - https://arxiv.org/abs/2411.04872
    - GSM8K
- **Positional Bias**
    - https://arxiv.org/abs/2410.14641
    - https://github.com/Rachum-thu/LongPiBench
- **'Reasoning'**
    - AGI Eval
        - https://arxiv.org/abs/2304.06364
    - BoolQ
        - https://arxiv.org/abs/1905.10044
    - Counterfactual Reasoning Assessment (CRASS)
        - https://arxiv.org/abs/2112.11941
    - Discrete Reasoning Over Paragraphs (DROP)
        - https://arxiv.org/abs/1903.00161
    - MMLU-Pro
        - https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        - https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main
    - Large-scale ReAding Comprehension Dataset From Examinations (RACE)
        - https://arxiv.org/abs/1704.04683
    - Physical Interaction: Question Answering (PIQA)
        - https://github.com/ybisk/ybisk.github.io/tree/master/piqa/data
    - Question Answering in Context (QuAC)
        - https://quac.ai/
- **Role Play**
    - **Conversation Relevancy**
        - DeepEval
            * `assesses whether your LLM chatbot is able to generate relevant responses throughout a conversation. It is calculated by looping through each turn individually and adopts a sliding window approach to take the last min(0, current turn number — window size) turns into account to determine whether it is relevant or not. The final conversation relevancy metric score is simply the number of turn responses that is relevant divided by the total number of turns in a conversational test case.` 
            * https://docs.confident-ai.com/docs/metrics-conversation-relevancy
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
            - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/role_adherence
        - General Research / Unsorted
            - https://arxiv.org/html/2406.00627v1
            - https://mp.weixin.qq.com/s/H2KNDGRNHktHiQc3sayFsA
            - https://mp.weixin.qq.com/s/2lbCMo64-nU5yRz1cLQxYA
            - https://mp.weixin.qq.com/s/E5qp5YPYPVaLM07OumDTRw 
            - https://mp.weixin.qq.com/s/yoM-srJYGGfyd1VXirg_Hg
        - RP-Bench
            - https://boson.ai/rpbench-blog/
        - PersonaGym
            - https://arxiv.org/abs/2407.18416
        - Collections of research
            - https://github.com/MiuLab/PersonaLLM-Survey
        - Notes
            - https://ianbicking.org/blog/2024/04/roleplaying-by-llm
    - **Knowledge Retention**
         - `chatbot is able to retain information presented to it throughout a conversation. It is calculated by first extracting a list of knowledges presented to it up to the certain turn in a conversation, and determining whether the LLM is asking for information that is already present in the turn response. The knowledge retention score is simply the number of turns without knowledge attritions divided by the total number of turns.`
         - https://docs.confident-ai.com/docs/metrics-knowledge-retention
    - **Conversation Completeness**
        - `chatbot is able to fulfill user requests throughout a conversation. It is useful because conversation completeness can be used as a proxy to measure user satisfaction and chatbot effectiveness. It is calculated by first using an LLM to extract a list of high level user intentions found in the conversation turns, before using the same LLM to determine whether each intention was met and/or satisfied throughout the conversation.`
        - https://docs.confident-ai.com/docs/metrics-conversation-completeness
- **Specific-Tasks**
    - CRMArena
        - https://arxiv.org/abs/2411.02305
- **Summarization**
    - Why use LLMs for summarization vs other approaches
        - https://www.mdpi.com/2673-4591/59/1/194
        - https://www.sciencedirect.com/science/article/pii/S2949719124000189
        - https://arxiv.org/pdf/2403.02901
    - Measuring LLM Summarization
        - https://stackoverflow.com/questions/9879276/how-do-i-evaluate-a-text-summarization-tool
    - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/summarization
    - https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task
    - https://arxiv.org/abs/2407.01370v1
    - https://arxiv.org/html/2403.19889v1
    - https://github.com/salesforce/summary-of-a-haystack
    - https://docs.cohere.com/page/summarization-evals
    - https://docs.cohere.com/page/grounded-summarization
    - Books
        - https://arxiv.org/abs/2404.01261
        - https://arxiv.org/abs/2310.00785
        - https://arxiv.org/abs/2407.06501
        - https://arxiv.org/abs/2205.09641
        - https://github.com/DISL-Lab/FineSurE-ACL24
        - https://arxiv.org/abs/2404.01261
        - https://openreview.net/forum?id=7Ttk3RzDeu
      - News Sources
          - https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News
    - ACI-Bench
        - https://arxiv.org/abs/2306.02022
    - DeepEval
        - https://docs.confident-ai.com/docs/metrics-summarization
    - MS Marco
        - https://arxiv.org/abs/1611.09268
    - Query-based Multi-domain Meeting Summarization (QMSum)
        - https://arxiv.org/abs/2104.05938
        - https://github.com/Yale-LILY/QMSum
    - RAGAS
        - https://arxiv.org/abs/2309.15217
        - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/summarization_score/
        - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/noise_sensitivity/
        - https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/agents/#topic_adherence
- **Text Comprehension**
- **QA (General)**
    * https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question
    * Multi-Genre Natural Language Inference (MultiNLI)
        * https://arxiv.org/abs/1704.05426
    * TriviaQA
        * https://arxiv.org/abs/1705.03551
        * https://github.com/mandarjoshi90/triviaqa
- **Tool Calling**
    - DeepEval
        - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/tool_correctness
- **Toxicity Testing**
    - DeepEval
        - https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/toxicity
    - HHH
        - https://arxiv.org/abs/2112.00861
        - https://github.com/anthropics/hh-rlhf
    - ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection
        - https://github.com/microsoft/TOXIGEN/tree/main
    - TruthfulQA
        - https://arxiv.org/abs/2109.07958v2
        - https://github.com/sylinrl/TruthfulQA
- **Vibes**
    - AidanBench
        - https://github.com/aidanmclaughlin/AidanBench
- **Links**
    - https://github.com/huggingface/evaluation-guidebook/blob/main/contents/automated-benchmarks/tips-and-tricks.md


----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="search-eval"></a> Search Evaluation
- **101**
  - F
- **Basic Search Eval via swyx**
  1. Take your existing text chunks
  2. Generate questions that could be answered by each chunk
  3. Store {question, chunk_id} pairs
  4. Test: Does your retrieval system find the source chunk?

Retrieval Granularity
    https://chentong0.github.io/factoid-wiki/
    https://github.com/chentong0/factoid-wiki
----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### <a name="rag-eval"></a> RAG Evaluation
- **101**
- **RAG Eval Plan:**
    - The generic idea however: you take a (full, unchunked) document and ask an LLM to generate a question with that document as well as give the factual answer to it. Enforce via prompts to make it use the document only and make it as hard as you want (eg. maybe sometimes you want it to consider 2 documents and make a question that uses bits of both). This gives you a ground truth dataset.
        * You then kick off your RAG pipeline on your documents. They will be chunked, indexed and stored. Then you fire all the questions of your ground truth set at your RAG pipeline and check 1. If it found chunks of the correct document and 2. Ask an LLM various evaluation questions about the generated answer vs. the ground truth answer ( like: how related are they, is there content in the answer that is not in the doc chunks, etc).
        * This gives you a good idea how well your retrieval (and with that, indexing) works, and how well your full pipeline works. As a bonus you could also keep track of which chunk(s) the ground truth answer was based on and use that for retrieval evaluation too. 
    - **When to move on from Primarily Synthetic Data to Real User Data**
        - 80% recall on synthetic tests
        - Good understanding of failure cases
        - Clear chunking strategy
        - Growing set of real user examples
- **Metrics**
    - 3 General Categories
       1. Retrieval Metric
       2. Generation-Specific Metric
       3. RAG-specific Metric
- 
- 
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
- **DataSets**
    - https://huggingface.co/datasets/enelpol/rag-mini-bioasq
    - https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia
    - RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems
        - https://arxiv.org/abs/2407.11005
        - https://huggingface.co/datasets/rungalileo/ragbench
- **Generating Synthetic Data**
    - https://www.turingpost.com/p/sytheticdata

- **RAG-Specific Tuning/Modfications**
    - **Pre-Training/Finetuning**
        - RankRAG
            - https://arxiv.org/abs/2407.02485v1
        - RAG-FiT
            - https://github.com/IntelLabs/RAG-FiT
        - RAG Foundry
            - https://arxiv.org/pdf/2408.02545
    - **Inference Time**
        * https://arxiv.org/pdf/2408.14906

- **RAG-Specific Models**
    - https://huggingface.co/lightblue/kurage-multilingual
    - https://arxiv.org/pdf/2407.14482

```
The fundamental idea of evaluating a retriever is to check how well the retrieved content matches the expected contents. For evaluating a RAG pipeline end to end, we need query and ground truth answer pairs. The ground truth answer must be grounded on some "ground" truth chunks. This is a search problem, it's easiest to start with tradiaional Information retrieval metrics.

You might already have access to such evaluation dataset depending on the nature of your application or you can synthetically build one. To build one you can retrieve random documents/chunks and ask an LLM to generate query-answer pairs - the underlying documents/chunks will act as your ground truth chunk.

Retriever Evaluation
We can evaluate a retriever using traditional ML metrics. We can also evaluate by using a powerful LLM (next section).

Below we are importing both traditional metrics and LLM as a judge metric from the scripts/retrieval_metrics.py file.

    Hit Rate: Measures the proportion of queries where the retriever successfully returns at least one relevant document.
    MRR (Mean Reciprocal Rank): Evaluates how quickly the retriever returns the first relevant document, based on the reciprocal of its rank.
    NDCG (Normalized Discounted Cumulative Gain): Assesses the quality of the ranked retrieval results, giving more importance to relevant documents appearing earlier.
    MAP (Mean Average Precision): Computes the mean precision across all relevant documents retrieved, considering the rank of each relevant document.
    Precision: Measures the ratio of relevant documents retrieved to the total documents retrieved by the retriever.
    Recall: Evaluates the ratio of relevant documents retrieved to the total relevant documents available for the query.
    F1 Score: The harmonic mean of precision and recall, providing a balance between both metrics to gauge retriever performance.
```

- **Evaluation Benchmarks**
    - KITE
        - https://github.com/D-Star-AI/KITE



Evaluating RAG Cohere
    https://docs.cohere.com/page/rag-evaluation-deep-dive#generation-evaluation

- **Frameworks**
    - RAGAS
        - https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/
        - https://docs.ragas.io/en/latest/concepts/testset_generation.html
    - Tonic
        - https://github.com/TonicAI/tonic_validate
    - RAGEval
        - https://arxiv.org/pdf/2408.01262
        - https://github.com/OpenBMB/RAGEval
    - AutoRAG
        - https://github.com/Marker-Inc-Korea/AutoRAG
    - Promptflow
        - https://github.com/microsoft/promptflow/tree/main/examples/flows/evaluation/eval-qna-rag-metrics
    - HELMET: How to Evaluate Long-context Language Models Effectively and Thoroughly
        - https://github.com/princeton-nlp/helmet


- Papers
  - https://arxiv.org/abs/2309.01431
  - https://arxiv.org/abs/2411.03538




----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="embeddings-retrieval-eval"></a> Embeddings Retrieval Evaluation


----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### <a name="vlm-evals"></a> VLM Evaluations
https://arxiv.org/abs/2411.04075
https://arxiv.org/abs/2411.02571
https://github.com/TRI-ML/vlm-evaluation
- xkcd bench: https://github.com/arnokha/explain-xkcd-with-llms

- Document Understanding
    - https://arxiv.org/html/2411.01106v1

----------------------------------------------------------------------------------------------------------------
