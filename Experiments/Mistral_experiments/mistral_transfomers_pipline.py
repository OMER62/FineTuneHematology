import json
from huggingface_hub import login
from dotenv import load_dotenv
import os
import datetime
import logging
import time
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, PromptTemplate,  get_response_synthesizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import NodeParser, SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine

from pinecone import Pinecone, ServerlessSpec
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.service_context import ServiceContext

# Load environment variables
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Check if the index exists, if not, create one
if 'hematology-index' not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name='hematology-index',
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Load the vector store index
pinecone_index = pc.Index(name="hematology-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

class SentenceTransformerEmbedModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        
    def get_text_embedding_batch(self, texts, **kwargs):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        embeddings = torch.cat([embeddings, embeddings], dim=1)  # Concatenate the embeddings to get 1536 dimensions
        return embeddings.cpu().numpy().tolist()
        
    def get_agg_embedding_from_queries(self, queries, **kwargs):
        embeddings = self.model.encode(queries, convert_to_tensor=True)
        agg_embedding = torch.mean(embeddings, dim=0)
        agg_embedding = torch.cat([agg_embedding, agg_embedding], dim=0)  # Concatenate the aggregated embedding to get 1536 dimensions
        return agg_embedding.cpu().numpy().tolist()

embed_model = SentenceTransformerEmbedModel("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Define the LLM model
llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    context_window=32000,
    generate_kwargs={"temperature": 0},
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
    device_map="auto",
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=10)

# Improve the prompt template
SYSTEM_PROMPT = """You are a medical doctor that specializes in Hematology. 
Return only the letter corresponding to the correct answer to the following question, 
do not reply using a complete sentence and only give the answer in the following format: '(x)':
"""
# for llama2
query_wrapper_prompt_llm = PromptTemplate(
    "<s>[INST] <<SYS>>\n" + SYSTEM_PROMPT + "\n<</SYS>>\n\n{query_str} [/INST]"
)
# for the pipeline query
query_wrapper_prompt_pipeline = PromptTemplate(
    "<s>[INST] <<SYS>>\n" + SYSTEM_PROMPT + "\n<</SYS>>\n\n{topic} [/INST]"
)
# prompt_tmpl = PromptTemplate(prompt_str)

summarizer = TreeSummarize(llm=llm)

reranker = CohereRerank()

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": query_wrapper_prompt_pipeline,
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

df = pd.read_excel("/home/koroli/test/Hematology-pdf/small_test.xlsx")


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

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
response_synthesizer = get_response_synthesizer(llm=llm, service_context=service_context)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)





# Run both processes and save the results

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def modelResponse_withoutRag(llm, df):
    questions = df['Questions']
    llm_responses = []
    elapsed_times = []  # List to store elapsed times for each query

    for q in questions:
        q = str(q)  # Ensure q is treated as a string
        start_time = time.time()  # Start time for each query
        # Generate answer using the LLM with the prompt template
        answer = llm.complete(SYSTEM_PROMPT).text
        end_time = time.time()  # End time for each query
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        llm_responses.append(answer)
        elapsed_times.append(elapsed_time)  # Append the elapsed time to the list

    df['MistralAI'] = llm_responses
    df['ElapsedTime'] = elapsed_times  # Add the elapsed times to the DataFrame
    return df

print("Start measuring time")
start = time.time()

new_df = modelResponse_withoutRag(llm, df)

print("End of LLM answers")
end = time.time()
print(f"Time for inference: {end - start}")

timestamp = time.strftime("%Y%m%d_%H%M%S")
new_df.to_csv(f"Mistral_Experiment_{timestamp}_llm_only.csv")
print(new_df)
# # Full pipeline run
# experimentResults_pipeline = run_rag_process(p, df)
# experimentResults_pipeline.to_csv(f"Experiment_{timestamp}_pipeline_transofrmersfinal3.csv")
# print(experimentResults_pipeline)

# # Simplified RAG run
# experimentResults_simple = modelResponse_withoutRag(llm, query_engine, df)
# experimentResults_simple.to_csv(f"Experiment_{timestamp}_simple_ragTransofrmersHarrisonTextBook.csv")
# print(experimentResults_simple)

print("The model finished answering all the questions for both processes.")
