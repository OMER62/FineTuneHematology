from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.postprocessor.cohere_rerank import CohereRerank
import pandas as pd
from dotenv import load_dotenv
import os

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.query_pipeline import QueryPipeline

# Load the vector store index
load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index(name="hematology-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# Define the LLM model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))
retriever = index.as_retriever(similarity_top_k=10)

prompt_str = "As a hematology specialist, please return the letter corresponding to the correct answer for the following scenario: {topic}. Provide only the letter of the correct option."

prompt_tmpl = PromptTemplate(prompt_str)

summarizer = TreeSummarize(llm=llm)
reranker = CohereRerank()  # Specify the number of top results to rerank

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": prompt_tmpl,
        "retriever": retriever,
        "summarizer": summarizer,
        "reranker": reranker,  # Add reranker to the pipeline
    }
)

p.add_link("input", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("input", "reranker", dest_key="query_str")
p.add_link("reranker", "summarizer", dest_key="nodes")
p.add_link("input", "summarizer", dest_key="query_str")
# Link reranker output to summarizer

df = pd.read_excel(
    "./American College of Physicians - MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021).xlsx")


def run_rag_process(p, df):
    questions = df['Questions']
    responses = []

    for question in questions:
        answer = p.run(topic=question)
        # Store the result
        responses.append({
            "Question": question,
            "Answer": answer
        })

    return responses


df2 = run_rag_process(p, df)
print(df2)
