# Chunking

## Overview
- Chunking is the process of breaking down a document into smaller pieces, or "chunks". This is useful for a variety of reasons, such as:
- 






### Types of Chunking


### Implementation in tldw
- 

```

Instead of chunking based on size, do it based on content. E.g: each table, chart, paragraph, column (for multi column layout pages). Vision LLM are pretty good now, for this task. For each chunk, also store summary in a full text index. This should improve R part of RAG

```

### Link Dump:
https://gleen.ai/blog/agentic-chunking-enhancing-rag-answers-for-completeness-and-accuracy/
https://github.com/carlosplanchon/betterhtmlchunking
https://github.com/segment-any-text/wtpsplit
https://github.com/Unsiloed-AI/Unsiloed-chunker
https://ai.gopubby.com/21-chunking-strategies-for-rag-f28e4382d399?gi=d29ab391014f
https://www.ibm.com/think/topics/agentic-chunking
https://www.reddit.com/r/Rag/comments/1ljhksy/best_chunking_strategy_for_the_medical_rag_system/
https://arxiv.org/abs/2506.16035
https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_double_merging_chunking/

https://github.com/speedyk-005/chunklet
