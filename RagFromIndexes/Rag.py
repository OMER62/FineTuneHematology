# https://github.com/edumunozsala/llamaindex-RAG-techniques/blob/main/query-pipelines.ipynb
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.postprocessor.cohere_rerank import CohereRerank
import pandas as pd
from dotenv import load_dotenv
import os
import datetime
import logging
import time

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.query_pipeline import QueryPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the vector store index
load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# First index setup (already in your code)
pc1 = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index1 = pc1.Index(name="hematology-index")
vector_store1 = PineconeVectorStore(pinecone_index=index1)
embed_model1 = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))
retriever1 = VectorStoreIndex.from_vector_store(vector_store=vector_store1, embed_model=embed_model1).as_retriever(
    similarity_top_k=10)

# Second index setup
pc2 = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))  # Assuming the same API key can be used
index2 = pc2.Index(name="medical-cases-index")
vector_store2 = PineconeVectorStore(pinecone_index=index2)
embed_model2 = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))
retriever2 = VectorStoreIndex.from_vector_store(vector_store=vector_store2, embed_model=embed_model2).as_retriever(
    similarity_top_k=10)

# Define the LLM model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=os.environ.get("OPENAI_API_KEY"))

# Improve the prompt template
prompt_str = "You are a student taking a hematology exam. For the following scenario, analyze the given options and return only the letter corresponding to the most appropriate answer: {topic}"
prompt_tmpl = PromptTemplate(prompt_str)

summarizer = TreeSummarize(llm=llm)

reranker = CohereRerank()

pipeline = QueryPipeline(verbose=True)
pipeline.add_modules({
    "input": prompt_tmpl,
    "retriever1": retriever1,
    "retriever2": retriever2,
    "reranker": reranker,
    "summarizer": summarizer,
})

# Set up the pipeline links
pipeline.add_link("input", "retriever1")
pipeline.add_link("input", "retriever2")
pipeline.add_link("retriever1", "reranker", dest_key="nodes")
pipeline.add_link("retriever2", "reranker", dest_key="nodes")
pipeline.add_link("input", "reranker", dest_key="query_str")
pipeline.add_link("input", "summarizer", dest_key="query_str")
pipeline.add_link("reranker", "summarizer", dest_key="nodes")

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
experimentResults = run_rag_process(pipeline, df)
experimentResults.to_csv(f"Experiment_{timestamp}.csv")
print(experimentResults)
