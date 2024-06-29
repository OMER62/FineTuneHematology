from llama_index.core.response_synthesizers import TreeSummarize
import pandas as pd
from dotenv import load_dotenv
import os
import datetime

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
llm = OpenAI(model="gpt-3.5-turbo", temperature=1, api_key=os.environ.get("OPENAI_API_KEY"))
retriever = index.as_retriever(similarity_top_k=5)

prompt_str = "As a hematology specialist, please respond to the following scenario by returning only the letter corresponding to the correct answer from the given options: {topic}"

prompt_tmpl = PromptTemplate(prompt_str)

summarizer = TreeSummarize(llm=llm)

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": prompt_tmpl,
        "retriever": retriever,
        "summarizer": summarizer,
    }
)

p.add_link("input", "retriever")
p.add_link("input", "summarizer", dest_key="query_str")
p.add_link("retriever", "summarizer", dest_key="nodes")

df = pd.read_excel(
    "./American College of Physicians - MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021).xlsx")


def run_rag_process(p, df):
    responses = []

    for row in df[:20].itertuples():
        question = row.Questions
        answer = row.Answers
        ragAnswer = p.run(topic=question)
        # Store the result
        responses.append({
            "Question": question,
            "Answer": answer,
            "AnswerWithRag": ragAnswer
        })

    return pd.DataFrame(responses)

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experimentResults = run_rag_process(p, df)
experimentResults.to_csv(f"Experiment_{timestamp}.csv")
print(experimentResults)
