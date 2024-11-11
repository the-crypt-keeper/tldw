# Evaluation Plans

----------------------------------------------------------------------------------------------------------------
## Table of Contents
- []()
- []()
- []()
- []()
- []()
----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### Introduction

----------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------
### Evaluation Plan
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




Demo Setup Process:
3. Add/Ingest Defcon 32 War Stories: https://www.youtube.com/playlist?list=PL9fPq3eQfaaDo7as-xxbY6r9ts86lxkwc
3. Add/Ingest Defcon 32 Villages talks:
   - RF Village: https://www.youtube.com/playlist?list=PLd8C1LGNy4UF-BhdT4583mV5xVPcHXMTw
   - VR Village: https://www.youtube.com/playlist?list=PLD2EdZEOJAKfa2qvqLHuLpyptg0Vnwtoq
   - Red Team Village: https://www.youtube.com/playlist?list=PLruly0ngXhPFIXuS_FwKi3vLUXvxz2-iv
   - Policy: https://www.youtube.com/playlist?list=PL9fPq3eQfaaBVQMmIp3EUhYsUpxXiwI63
   - Video Team: https://www.youtube.com/playlist?list=PL9fPq3eQfaaD_2THtBVamFKcNv-5Bl8xP