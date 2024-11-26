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

PDF Parsing
    https://github.com/pdf2htmlEX/pdf2htmlEX
    https://camelot-py.readthedocs.io/en/master/
    https://github.com/kermitt2/grobid
    https://arxiv.org/html/2410.09871v1#S6
    https://github.com/Filimoa/open-parse/
    https://github.com/nlmatics/nlm-ingestor
    https://github.com/conjuncts/gmft
    https://github.com/xavctn/img2table
    Benchmarks
        https://github.com/py-pdf/benchmarks
    Pipeline
        https://ai.gopubby.com/demystifying-pdf-parsing-02-pipeline-based-method-82619dbcbddf
        https://ai.gopubby.com/demystifying-pdf-parsing-04-ocr-free-large-multimodal-model-based-method-0fdab50db048
        https://pub.towardsai.net/demystifying-pdf-parsing-05-unifying-separate-tasks-into-a-small-model-d3739db021f7
        https://ai.gopubby.com/demystifying-pdf-parsing-06-representative-industry-solutions-5d4a1cfe311b

ETL
    https://arxiv.org/abs/2410.12189
    https://ucbepic.github.io/docetl/concepts/optimization/
    https://arxiv.org/abs/2410.21169

UX
    https://arxiv.org/abs/2410.22370
    https://archive.is/h8w61
	https://uxdesign.cc/how-the-right-ux-metrics-show-game-changing-value-9d95c46b0479?sk=2b9d400046e4484f858a409cc56b355a&utm_source=substack&utm_medium=email
	https://uxdesign.cc/embracing-playing-as-the-core-of-design-92215f031bed?utm_source=substack&utm_medium=email


Import HTML Files
    
Manga
https://github.com/ragavsachdeva/magi
Emails:
https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/email.py
ODT, Docx, csv
    pypandoc
Evernote:
    https://help.evernote.com/hc/en-us/articles/209005557-Export-notes-and-notebooks-as-ENEX-or-HTML
Excel:
    https://ajac-zero.com/posts/how-to-create-accurate-fast-rag-with-excel-files/
    https://github.com/ajac-zero/excel-rag-example
    https://github.com/sivadhulipala1999/SheetSimplify_with_RAG
CSV
    CSV -> HTML https://blog.finxter.com/5-best-ways-to-convert-csv-to-html-in-python/
    https://github.com/jazzband/tablib
CHM:
    https://github.com/dottedmag/archmage



URL Shortener
    https://realpython.com/fastapi-python-web-apis/
    https://realpython.com/build-a-python-url-shortener-with-fastapi/



Flashcard prompt
Fantastic-End9783@reddit
```
I want you to create comprehensive and easy to consume Anki flashcards. They must thoroughly cover REPLACEWITHLECTURE
Do not do this until you I ask you after you  output the following request .
Thoroughly go through the provided sources and divide them into parts. This is so you can compressively cover the lecture and not skip information. The parts do not have to have the exact same split of questions made for them.
Please list the overall topic of the part and then their sub-topics.
Here is an example:
Part 1: (Insert title here)
- (Topic 1)
- (Topic 2)
```
```
Create comprehensive and easy to consume Anki flashcards for Part 1 of REPLACEWITHLECTURE. If there is embedded questions or topics per question, split them up into their own individual questions. Start each question on a new line. 
Use the following as a scaffold for the expected output:
What noise does a cow make? ; Moo
What is the capital of Australia? ; Canberra
```


#1: `Do not start writing yet, First explain everything I wanted you to do in this Prompt in Detail?`
#2: `I need this written in human tone. Humans have fun when they write — robots don’t. Chat GPT, engagement is the highest priority. Be conversational, empathetic, and occasionally humorous. Use idioms, metaphors, anecdotes, and natural dialogue.`
#3: `Before you answer, I want you to ask me all the missing information that I didn’t provide but it will help you better understand my needs and the specific output I want.`
#4: `Criticize yourself/your answer.`
#5: `Why did you write what you wrote? Give me all the reasons, Plus I want a full detailed analysis and breakdown of everything in a tabular format. Also add How could this be made better. Use my prompt as reference to further clarify the ‘Why’.`
#6: `Before you answer this, Highlight 20 potential risks or blind spots I might not have considered based on my request.`
#7: `Identify areas in this article where examples, analogies, or case studies would improve understanding.`


Jailbreaks
    https://arxiv.org/abs/2311.16119
    https://www.lakera.ai/blog/visual-prompt-injections
    https://github.com/0xk1h0/ChatGPT_DAN
    https://arxiv.org/pdf/2409.17458
    https://github.com/verazuo/jailbreak_llms
    https://arxiv.org/pdf/2308.03825
    https://arxiv.org/pdf/2409.00137
    https://arxiv.org/pdf/2409.11445


