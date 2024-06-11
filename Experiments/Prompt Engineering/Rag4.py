import pandas as pd
import os
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.query_pipeline import QueryPipeline

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
pinecone_key = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)
pinecone_index = pc.Index(name="hematology-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Embedding and LLM setup
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=api_key)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

# Retrieval and summarization setup
retriever = index.as_retriever(similarity_top_k=10)
summarizer = TreeSummarize(llm=llm)

# Reranking setup
reranker = CohereRerank()

# Define a more specific prompt
prompt_str = "As a hematology expert, analyze the scenario: '{topic}'. Provide the best answer choice (A, B, C, D, E) based on medical knowledge."

prompt_tmpl = PromptTemplate(prompt_str)

# Pipeline setup
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": prompt_tmpl,
        "retriever": retriever,
        "summarizer": summarizer,
        "reranker": reranker
    }
)

# Link pipeline stages
p.add_link("input", "retriever")
p.add_link("retriever", "summarizer", src_key="nodes", dest_key="nodes")
p.add_link("summarizer", "reranker", src_key="responses", dest_key="responses")
p.add_link("reranker", "output", src_key="responses", dest_key="final_output")

# Load the data
df = pd.read_excel(
    "./American College of Physicians - MKSAP 19_ medical knowledge self-assessment program. Hematology-American College of Physicians (2021).xlsx")

def run_rag_process(pipeline, df):
    responses = []

    for row in df.itertuples():
        question = row.Questions
        answer = row.Answers
        # Run the pipeline with the question
        processed_answer = pipeline.run(topic=question)

        # Store the result
        responses.append({
            "Question": question,
            "Answer": answer,
            "AnswerWithRag": processed_answer
        })

    return pd.DataFrame(responses)

# Process questions and save results
df_results = run_rag_process(p, df)
df_results.to_csv("Updated_Results.csv")
print(df_results)
