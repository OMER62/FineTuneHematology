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
pinecone_index = pc.Index(name="hematology-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# Define the LLM model
llm = OpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"))
retriever = index.as_retriever(similarity_top_k=5)  # Increase the number of retrieved documents

# Improve the prompt template
prompt_str = "You are a student taking a hematology exam. For the following scenario, analyze the given options and return only the letter corresponding to the most appropriate answer: {topic}"
prompt_tmpl = PromptTemplate(prompt_str)

summarizer = TreeSummarize(llm=llm)

reranker = CohereRerank()

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": prompt_tmpl,
        "retriever": retriever,
        "reranker": reranker,
        "summarizer": summarizer,
    }
)
p.add_link("input", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("input", "reranker", dest_key="query_str")
p.add_link("input", "summarizer", dest_key="query_str")
p.add_link("reranker", "summarizer", dest_key="nodes")

df = pd.read_excel(
    "./American College of Physicians - MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021).xlsx")


def run_rag_process(p, df):
    responses = []

    i = 0  # Initialize the counter outside the loop
    while i < len(df):  # Continue looping as long as i is less than the length of the DataFrame
        row = df.iloc[i]  # Access the row by index
        question = row['Questions']  # Access columns by name
        answer = row['Answers']
        try:
            ragAnswer = p.run(topic=question)
            # Store the result
            responses.append({
                "Question": question,
                "Answer": answer,
                "AnswerWithRag": ragAnswer
            })
            i += 1
        except Exception as e:
            logging.error(f"Error processing question: {question}")
            logging.error(str(e))
            time.sleep(5)
    return pd.DataFrame(responses)


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experimentResults = run_rag_process(p, df)
experimentResults.to_csv(f"Experiment_{timestamp}.csv")
print(experimentResults)
