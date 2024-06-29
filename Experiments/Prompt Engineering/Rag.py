import pandas as pd
from dotenv import load_dotenv
import os

from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate, PromptHelper, BasePromptTemplate
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import get_response_synthesizer

# Load the vector store index
load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index(name="hematology-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# Define the LLM model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))

# Define the postprocessor and response synthesizer
reranker = CohereRerank()

retriever = index.as_retriever(similarity_top_k=10)
# try chaining basic prompts

prompt_str = "As a hematology specialist, please respond to the following scenario by returning only the letter corresponding to the correct answer from the given options: {topic}"

prompt_tmpl = PromptTemplate(prompt_str)


summarizer = TreeSummarize(llm=llm)
p = QueryPipeline( verbose=True)

# Add modules
p.add_modules(
    {
        "prompt_tmpl": prompt_tmpl,
        "llm": llm,
        "retriever": retriever,
        "summarizer": summarizer,
        "reranker": reranker,
    }
)

# Correctly set the links
p.add_link("prompt_tmpl", "llm")
p.add_link("llm", "retriever")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("llm", "reranker", dest_key="query_str")
p.add_link("reranker", "summarizer", dest_key="nodes")
p.add_link("llm", "summarizer", dest_key="query_str")

# Ensure only 'prompt_tmpl' is the root
# Run the pipeline
query = "A S8-year-old man is evaluated for possible smoldering myeloma. Medical history is unremarkable, and he takes no medications. On physical examination, vital signs and other exam- ination flndings are normal. Serum protein electrophoresis and immunoflxation show an IgA protein spike of 3.5 g/dl (35 g/L). Bone mar- row biopsy reveals 50% clonal plasma cells. Whole-body low-dose CT scan is negative for bone lesions. Which of the following is the most appropriate imaging test to perform next? (A) Bone scan (B) Skeletal survey (C) Whole-bodyMRI (D) No further testing"
output = p.run(topic=query)
print(output)

