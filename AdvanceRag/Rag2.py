# https://github.com/edumunozsala/llamaindex-RAG-techniques/blob/main/query-pipelines.ipynb
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
import pandas as pd
from dotenv import load_dotenv
import os
import datetime
import logging
import time

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate, load_index_from_storage, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.query_pipeline import QueryPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the vector store index
load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index(name="medical-cases-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Define the LLM model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=os.environ.get("OPENAI_API_KEY"))
retriever = index.as_retriever(similarity_top_k=10)
# vector_index_chunk = load_index_from_storage(storage_context)

# Improve the prompt template
prompt_str = "You are a student taking a hematology exam. For the following scenario, analyze the given options and return only the letter corresponding to the most appropriate answer: {topic}"
prompt_tmpl = PromptTemplate(prompt_str)

bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=5)

summarizer = TreeSummarize(llm=llm)

reranker = CohereRerank()

lim_reorder = LongContextReorder()
# define query pipeline
pipe_5 = QueryPipeline(verbose=True)
pipe_5.add_modules(
    {
        "input": prompt_tmpl,
        "retriever": retriever,
        "bm25": bm25_retriever,
        "longcontext": lim_reorder,
        "reranker": reranker,
        "summarizer": summarizer,
    }
)

pipe_5.add_link("input", "retriever")
pipe_5.add_link("input", "bm25")
pipe_5.add_link("retriever", "longcontext", dest_key="nodes")
pipe_5.add_link("bm25", "longcontext", dest_key="nodes")
pipe_5.add_link("input", "longcontext", dest_key='query_str')
pipe_5.add_link("longcontext", "reranker", dest_key="nodes")
pipe_5.add_link("input", "reranker", dest_key="query_str")
pipe_5.add_link("reranker", "summarizer", dest_key="nodes")
pipe_5.add_link("input", "summarizer", dest_key="query_str")

df = pd.read_excel(
    "./American College of Physicians - MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021).xlsx")


def run_rag_process(p, df):
    responses = []

    for row in df.itertuples():
        question = row.Questions
        answer = row.Answers
        try:
            ragAnswer = p.run(topic=question)
            # Store the result
            responses.append({
                "Question": question,
                "Answer": answer,
                "AnswerWithRag": ragAnswer
            })
        except Exception as e:
            logging.error(f"Error processing question: {question}")
            logging.error(str(e))
        time.sleep(5)
    return pd.DataFrame(responses)


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experimentResults = run_rag_process(p, df)
experimentResults.to_csv(f"Experiment_{timestamp}.csv")
print(experimentResults)
