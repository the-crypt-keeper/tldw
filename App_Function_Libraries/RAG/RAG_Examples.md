
```
##################################################################################################################
# RAG Pipeline 1
# 0.62    0.61    0.75    63402.0
# from langchain_openai import ChatOpenAI
#
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
#
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.storage import InMemoryStore
# import os
# from operator import itemgetter
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
# from langchain.retrievers import MergerRetriever
# from langchain.retrievers.document_compressors import DocumentCompressorPipeline


# def rag_pipeline():
#     try:
#         def format_docs(docs):
#             return "\n".join(doc.page_content for doc in docs)
#
#         llm = ChatOpenAI(model='gpt-4o-mini')
#
#         loader = WebBaseLoader('https://en.wikipedia.org/wiki/European_debt_crisis')
#         docs = loader.load()
#
#         embedding = OpenAIEmbeddings(model='text-embedding-3-large')
#
#         splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
#         splits = splitter.split_documents(docs)
#         c = Chroma.from_documents(documents=splits, embedding=embedding,
#                                   collection_name='testindex-ragbuilder-1724657573', )
#         retrievers = []
#         retriever = c.as_retriever(search_type='mmr', search_kwargs={'k': 10})
#         retrievers.append(retriever)
#         retriever = BM25Retriever.from_documents(docs)
#         retrievers.append(retriever)
#
#         parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=600)
#         splits = parent_splitter.split_documents(docs)
#         store = InMemoryStore()
#         retriever = ParentDocumentRetriever(vectorstore=c, docstore=store, child_splitter=splitter,
#                                             parent_splitter=parent_splitter)
#         retriever.add_documents(docs)
#         retrievers.append(retriever)
#         retriever = MergerRetriever(retrievers=retrievers)
#         prompt = hub.pull("rlm/rag-prompt")
#         rag_chain = (
#             RunnableParallel(context=retriever, question=RunnablePassthrough())
#             .assign(context=itemgetter("context") | RunnableLambda(format_docs))
#             .assign(answer=prompt | llm | StrOutputParser())
#             .pick(["answer", "context"]))
#         return rag_chain
#     except Exception as e:
#         print(f"An error occurred: {e}")


# To get the answer and context, use the following code
# res=rag_pipeline().invoke("your prompt here")
# print(res["answer"])
# print(res["context"])

############################################################################################################


############################################################################################################
# RAG Pipeline 2

# 0.6     0.73    0.68    3125.0
# from langchain_openai import ChatOpenAI
#
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.storage import InMemoryStore
# from langchain_community.document_transformers import EmbeddingsRedundantFilter
# from langchain.retrievers.document_compressors import LLMChainFilter
# from langchain.retrievers.document_compressors import EmbeddingsFilter
# from langchain.retrievers import ContextualCompressionRetriever
# import os
# from operator import itemgetter
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
# from langchain.retrievers import MergerRetriever
# from langchain.retrievers.document_compressors import DocumentCompressorPipeline


# def rag_pipeline():
#     try:
#         def format_docs(docs):
#             return "\n".join(doc.page_content for doc in docs)
#
#         llm = ChatOpenAI(model='gpt-4o-mini')
#
#         loader = WebBaseLoader('https://en.wikipedia.org/wiki/European_debt_crisis')
#         docs = loader.load()
#
#         embedding = OpenAIEmbeddings(model='text-embedding-3-large')
#
#         splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
#         splits = splitter.split_documents(docs)
#         c = Chroma.from_documents(documents=splits, embedding=embedding,
#                                   collection_name='testindex-ragbuilder-1724650962', )
#         retrievers = []
#         retriever = MultiQueryRetriever.from_llm(c.as_retriever(search_type='similarity', search_kwargs={'k': 10}),
#                                                  llm=llm)
#         retrievers.append(retriever)
#
#         parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=600)
#         splits = parent_splitter.split_documents(docs)
#         store = InMemoryStore()
#         retriever = ParentDocumentRetriever(vectorstore=c, docstore=store, child_splitter=splitter,
#                                             parent_splitter=parent_splitter)
#         retriever.add_documents(docs)
#         retrievers.append(retriever)
#         retriever = MergerRetriever(retrievers=retrievers)
#         arr_comp = []
#         arr_comp.append(EmbeddingsRedundantFilter(embeddings=embedding))
#         arr_comp.append(LLMChainFilter.from_llm(llm))
#         pipeline_compressor = DocumentCompressorPipeline(transformers=arr_comp)
#         retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=pipeline_compressor)
#         prompt = hub.pull("rlm/rag-prompt")
#         rag_chain = (
#             RunnableParallel(context=retriever, question=RunnablePassthrough())
#             .assign(context=itemgetter("context") | RunnableLambda(format_docs))
#             .assign(answer=prompt | llm | StrOutputParser())
#             .pick(["answer", "context"]))
#         return rag_chain
#     except Exception as e:
#         print(f"An error occurred: {e}")


# To get the answer and context, use the following code
# res=rag_pipeline().invoke("your prompt here")
# print(res["answer"])
# print(res["context"])

#
#
#
############################################################################################################
# Plain bm25 retriever
# class BM25Retriever(BaseRetriever):
#     """`BM25` retriever without Elasticsearch."""
#
#     vectorizer: Any
#     """ BM25 vectorizer."""
#     docs: List[Document] = Field(repr=False)
#     """ List of documents."""
#     k: int = 4
#     """ Number of documents to return."""
#     preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
#     """ Preprocessing function to use on the text before BM25 vectorization."""
#
#     class Config:
#         arbitrary_types_allowed = True
#
#     @classmethod
#     def from_texts(
#         cls,
#         texts: Iterable[str],
#         metadatas: Optional[Iterable[dict]] = None,
#         bm25_params: Optional[Dict[str, Any]] = None,
#         preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
#         **kwargs: Any,
#     ) -> BM25Retriever:
#         """
#         Create a BM25Retriever from a list of texts.
#         Args:
#             texts: A list of texts to vectorize.
#             metadatas: A list of metadata dicts to associate with each text.
#             bm25_params: Parameters to pass to the BM25 vectorizer.
#             preprocess_func: A function to preprocess each text before vectorization.
#             **kwargs: Any other arguments to pass to the retriever.
#
#         Returns:
#             A BM25Retriever instance.
#         """
#         try:
#             from rank_bm25 import BM25Okapi
#         except ImportError:
#             raise ImportError(
#                 "Could not import rank_bm25, please install with `pip install "
#                 "rank_bm25`."
#             )
#
#         texts_processed = [preprocess_func(t) for t in texts]
#         bm25_params = bm25_params or {}
#         vectorizer = BM25Okapi(texts_processed, **bm25_params)
#         metadatas = metadatas or ({} for _ in texts)
#         docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
#         return cls(
#             vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
#         )
#
#     @classmethod
#     def from_documents(
#         cls,
#         documents: Iterable[Document],
#         *,
#         bm25_params: Optional[Dict[str, Any]] = None,
#         preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
#         **kwargs: Any,
#     ) -> BM25Retriever:
#         """
#         Create a BM25Retriever from a list of Documents.
#         Args:
#             documents: A list of Documents to vectorize.
#             bm25_params: Parameters to pass to the BM25 vectorizer.
#             preprocess_func: A function to preprocess each text before vectorization.
#             **kwargs: Any other arguments to pass to the retriever.
#
#         Returns:
#             A BM25Retriever instance.
#         """
#         texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
#         return cls.from_texts(
#             texts=texts,
#             bm25_params=bm25_params,
#             metadatas=metadatas,
#             preprocess_func=preprocess_func,
#             **kwargs,
#         )
#
#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         processed_query = self.preprocess_func(query)
#         return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
#         return return_docs
############################################################################################################

############################################################################################################
# ElasticSearch BM25 Retriever
# class ElasticSearchBM25Retriever(BaseRetriever):
#     """`Elasticsearch` retriever that uses `BM25`.
#
#     To connect to an Elasticsearch instance that requires login credentials,
#     including Elastic Cloud, use the Elasticsearch URL format
#     https://username:password@es_host:9243. For example, to connect to Elastic
#     Cloud, create the Elasticsearch URL with the required authentication details and
#     pass it to the ElasticVectorSearch constructor as the named parameter
#     elasticsearch_url.
#
#     You can obtain your Elastic Cloud URL and login credentials by logging in to the
#     Elastic Cloud console at https://cloud.elastic.co, selecting your deployment, and
#     navigating to the "Deployments" page.
#
#     To obtain your Elastic Cloud password for the default "elastic" user:
#
#     1. Log in to the Elastic Cloud console at https://cloud.elastic.co
#     2. Go to "Security" > "Users"
#     3. Locate the "elastic" user and click "Edit"
#     4. Click "Reset password"
#     5. Follow the prompts to reset the password
#
#     The format for Elastic Cloud URLs is
#     https://username:password@cluster_id.region_id.gcp.cloud.es.io:9243.
#     """
#
#     client: Any
#     """Elasticsearch client."""
#     index_name: str
#     """Name of the index to use in Elasticsearch."""
#
#     @classmethod
#     def create(
#         cls, elasticsearch_url: str, index_name: str, k1: float = 2.0, b: float = 0.75
#     ) -> ElasticSearchBM25Retriever:
#         """
#         Create a ElasticSearchBM25Retriever from a list of texts.
#
#         Args:
#             elasticsearch_url: URL of the Elasticsearch instance to connect to.
#             index_name: Name of the index to use in Elasticsearch.
#             k1: BM25 parameter k1.
#             b: BM25 parameter b.
#
#         Returns:
#
#         """
#         from elasticsearch import Elasticsearch
#
#         # Create an Elasticsearch client instance
#         es = Elasticsearch(elasticsearch_url)
#
#         # Define the index settings and mappings
#         settings = {
#             "analysis": {"analyzer": {"default": {"type": "standard"}}},
#             "similarity": {
#                 "custom_bm25": {
#                     "type": "BM25",
#                     "k1": k1,
#                     "b": b,
#                 }
#             },
#         }
#         mappings = {
#             "properties": {
#                 "content": {
#                     "type": "text",
#                     "similarity": "custom_bm25",  # Use the custom BM25 similarity
#                 }
#             }
#         }
#
#         # Create the index with the specified settings and mappings
#         es.indices.create(index=index_name, mappings=mappings, settings=settings)
#         return cls(client=es, index_name=index_name)
#
#     def add_texts(
#         self,
#         texts: Iterable[str],
#         refresh_indices: bool = True,
#     ) -> List[str]:
#         """Run more texts through the embeddings and add to the retriever.
#
#         Args:
#             texts: Iterable of strings to add to the retriever.
#             refresh_indices: bool to refresh ElasticSearch indices
#
#         Returns:
#             List of ids from adding the texts into the retriever.
#         """
#         try:
#             from elasticsearch.helpers import bulk
#         except ImportError:
#             raise ImportError(
#                 "Could not import elasticsearch python package. "
#                 "Please install it with `pip install elasticsearch`."
#             )
#         requests = []
#         ids = []
#         for i, text in enumerate(texts):
#             _id = str(uuid.uuid4())
#             request = {
#                 "_op_type": "index",
#                 "_index": self.index_name,
#                 "content": text,
#                 "_id": _id,
#             }
#             ids.append(_id)
#             requests.append(request)
#         bulk(self.client, requests)
#
#         if refresh_indices:
#             self.client.indices.refresh(index=self.index_name)
#         return ids
#
#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         query_dict = {"query": {"match": {"content": query}}}
#         res = self.client.search(index=self.index_name, body=query_dict)
#
#         docs = []
#         for r in res["hits"]["hits"]:
#             docs.append(Document(page_content=r["_source"]["content"]))
#         return docs
############################################################################################################


############################################################################################################
# Multi Query Retriever
# class MultiQueryRetriever(BaseRetriever):
#     """Given a query, use an LLM to write a set of queries.
#
#     Retrieve docs for each query. Return the unique union of all retrieved docs.
#     """
#
#     retriever: BaseRetriever
#     llm_chain: Runnable
#     verbose: bool = True
#     parser_key: str = "lines"
#     """DEPRECATED. parser_key is no longer used and should not be specified."""
#     include_original: bool = False
#     """Whether to include the original query in the list of generated queries."""
#
#     @classmethod
#     def from_llm(
#         cls,
#         retriever: BaseRetriever,
#         llm: BaseLanguageModel,
#         prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
#         parser_key: Optional[str] = None,
#         include_original: bool = False,
#     ) -> "MultiQueryRetriever":
#         """Initialize from llm using default template.
#
#         Args:
#             retriever: retriever to query documents from
#             llm: llm for query generation using DEFAULT_QUERY_PROMPT
#             prompt: The prompt which aims to generate several different versions
#                 of the given user query
#             include_original: Whether to include the original query in the list of
#                 generated queries.
#
#         Returns:
#             MultiQueryRetriever
#         """
#         output_parser = LineListOutputParser()
#         llm_chain = prompt | llm | output_parser
#         return cls(
#             retriever=retriever,
#             llm_chain=llm_chain,
#             include_original=include_original,
#         )
#
#     async def _aget_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: AsyncCallbackManagerForRetrieverRun,
#     ) -> List[Document]:
#         """Get relevant documents given a user query.
#
#         Args:
#             query: user query
#
#         Returns:
#             Unique union of relevant documents from all generated queries
#         """
#         queries = await self.agenerate_queries(query, run_manager)
#         if self.include_original:
#             queries.append(query)
#         documents = await self.aretrieve_documents(queries, run_manager)
#         return self.unique_union(documents)
#
#     async def agenerate_queries(
#         self, question: str, run_manager: AsyncCallbackManagerForRetrieverRun
#     ) -> List[str]:
#         """Generate queries based upon user input.
#
#         Args:
#             question: user query
#
#         Returns:
#             List of LLM generated queries that are similar to the user input
#         """
#         response = await self.llm_chain.ainvoke(
#             {"question": question}, config={"callbacks": run_manager.get_child()}
#         )
#         if isinstance(self.llm_chain, LLMChain):
#             lines = response["text"]
#         else:
#             lines = response
#         if self.verbose:
#             logger.info(f"Generated queries: {lines}")
#         return lines
#
#     async def aretrieve_documents(
#         self, queries: List[str], run_manager: AsyncCallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         """Run all LLM generated queries.
#
#         Args:
#             queries: query list
#
#         Returns:
#             List of retrieved Documents
#         """
#         document_lists = await asyncio.gather(
#             *(
#                 self.retriever.ainvoke(
#                     query, config={"callbacks": run_manager.get_child()}
#                 )
#                 for query in queries
#             )
#         )
#         return [doc for docs in document_lists for doc in docs]
#
#     def _get_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun,
#     ) -> List[Document]:
#         """Get relevant documents given a user query.
#
#         Args:
#             query: user query
#
#         Returns:
#             Unique union of relevant documents from all generated queries
#         """
#         queries = self.generate_queries(query, run_manager)
#         if self.include_original:
#             queries.append(query)
#         documents = self.retrieve_documents(queries, run_manager)
#         return self.unique_union(documents)
#
#     def generate_queries(
#         self, question: str, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[str]:
#         """Generate queries based upon user input.
#
#         Args:
#             question: user query
#
#         Returns:
#             List of LLM generated queries that are similar to the user input
#         """
#         response = self.llm_chain.invoke(
#             {"question": question}, config={"callbacks": run_manager.get_child()}
#         )
#         if isinstance(self.llm_chain, LLMChain):
#             lines = response["text"]
#         else:
#             lines = response
#         if self.verbose:
#             logger.info(f"Generated queries: {lines}")
#         return lines
#
#     def retrieve_documents(
#         self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         """Run all LLM generated queries.
#
#         Args:
#             queries: query list
#
#         Returns:
#             List of retrieved Documents
#         """
#         documents = []
#         for query in queries:
#             docs = self.retriever.invoke(
#                 query, config={"callbacks": run_manager.get_child()}
#             )
#             documents.extend(docs)
#         return documents
#
#     def unique_union(self, documents: List[Document]) -> List[Document]:
#         """Get unique Documents.
#
#         Args:
#             documents: List of retrieved Documents
#
#         Returns:
#             List of unique retrieved Documents
#         """
#         return _unique_documents(documents)
############################################################################################################
```