# ISSUES

### Documented Issues:
1. FFmpeg is missing
2. cudnn8dlops.dll or whatever is missing/not in PATH
   * https://stackoverflow.com/questions/66083545/could-not-load-dynamic-library-cudnn64-8-dll-dlerror-cudnn64-8-dll-not-found

Create a blog post
    tldwproject.com
    https://www.open-notebook.ai/features/podcast.html

HF Embedding API - https://huggingface.co/BAAI/bge-large-en-v1.5
    Customize to use that for HF Demo

Help Llama.cpp
    -The most important tests are those in tests/ggml-backend-ops.cpp but those are also already in a comparatively good state; if you can find cases that are not covered adding them would be useful though.
    -Any tests of components that deal with memory management
    -The llama.cpp HTTP server currently has Python tests that call the API - can be improved
    -llama-bench - add an option for determining the performance for each token position instead of the average of all tokens in the range.
    -better error messages for failure modes of scripts/compare-llama-bench.py.
    -Scripts for submitting common language model benchmarks to the server also allow for a comparison with other projects.
        When academics introduce new techniques or quantization formats they usually measure the impact on quality using benchmarks like MMLU.
        llama.cpp currently has no standardized procedure for evaluating models on these benchmarks so a comparison of llama.cpp quality with academia is difficult.
        A standardized way for running benchmarks would also make it possible to compare the impact of things like different prompting techniques and quantization, especially between models where perplexity comparisons don't make sense.
        So you could for example better determine whether a heavily quantized 123b model or a less quantized 70b model performs better.
    -a simple Python script that does a performance benchmark for batched inference on the server
    -Better methods for evaluating samplers
        -A script that can be used to evaluate the performance of the sampler on a given dataset
        -Blind testing ala ChatArena where users are given two completions for a task and asked to choose the better one, created from two different random presets.
        -Use a model to generate multiple completions for a task with different seeds. Then rate/judge teh completions in terms of quality & diversity
        -Establish which sampling methods incur the smallest quality loss for a given target diversity.


Prompts
    Dspy
        https://pub.towardsai.net/dspy-machine-learning-attitude-towards-llm-prompting-0d45056fd9b7
        https://pub.towardsai.net/dspy-venture-into-automatic-prompt-optimization-d37b892091be
    https://github.com/f/awesome-chatgpt-prompts
    https://www.reddit.com/r/ChatGPTPro/comments/1g0aetb/turn_meeting_transcripts_into_valuable_insights/
    https://github.com/MaxsPrompts/Marketing-Prompts
    https://github.com/MIATECHPARTNERS/PromptChains
    https://github.com/zacfrulloni/Prompt-Engineering-Holy-Grail


LLMs.txt
    https://directory.llmstxt.cloud/


Logits
    https://github.com/NVIDIA/logits-processor-zoo




