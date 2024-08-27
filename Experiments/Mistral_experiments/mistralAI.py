import dotenv
import os
import pandas as pd
import time
from huggingface_hub import login

# access HuggingFace to get access to the LLMs
access_token = os.environ.get("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")
login(token=access_token)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.service_context import ServiceContext
from llama_index.core.node_parser import NodeParser, SimpleNodeParser
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings


# get the Hematology-index created by gpt-3.5 embeddings
pc = Pinecone(api_key="PINECONE_API")
pinecone_index = pc.Index(name="hematology-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
class SentenceTransformerEmbedModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def get_text_embedding_batch(self, texts, **kwargs):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.repeat(1, 2)  # Repeat the embeddings to get 1536 dimensions
        return embeddings.cpu().numpy().tolist()
    
    def get_agg_embedding_from_queries(self, queries, **kwargs):
        embeddings = self.model.encode(queries, convert_to_tensor=True)
        agg_embedding = torch.mean(embeddings, dim=0)
        return agg_embedding.cpu().numpy().tolist()

embed_model = SentenceTransformerEmbedModel("sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# configure a vector retriever
retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )
# create a system prompt
prompt_str = "As a hematology specialist, please respond to the following scenario by returning only the letter corresponding to the correct answer from the given options:"
query_wrapper_prompt = PromptTemplate(
    "<s>[INST]\n" + prompt_str + "[/INST]\n\n{query_str}</s> "
)

llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    context_window=32000,
    generate_kwargs={"temperature": 0},
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
    query_wrapper_prompt=query_wrapper_prompt,
    device_map="auto",
)
Settings.llm = llm
node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=200)
class SentenceTransformerEmbedModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def get_text_embedding_batch(self, texts, **kwargs):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.repeat(1, 2)  # Repeat the embeddings to get 1536 dimensions
        return embeddings.cpu().numpy().tolist()
    
    def get_agg_embedding_from_queries(self, queries, **kwargs):
        embeddings = self.model.encode(queries, convert_to_tensor=True)
        agg_embedding = torch.mean(embeddings, dim=0)
        return agg_embedding.cpu().numpy().tolist()

embed_model = SentenceTransformerEmbedModel("sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)




response_synthesizer = get_response_synthesizer(llm=llm, service_context=service_context)
df = pd.read_excel("/home/koroli/test/Hematology-pdf/Harrisons Hematology and Oncology 2ed_questionsPage743.xlsx")

query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

def modelResponse(llm,query_engine,df):
        questions=df['Questions']
        llmResponse=[]
        query_engineResponse=[]
        for q in questions:
            # LLM answer without RAG
            answer = llm.complete(q).text
            llmResponse.append(answer)
            # LLM answer with RAG
            query_answer = query_engine.query(q).response
            query_engineResponse.append(query_answer)

        df['MistralAI'] = llmResponse
        df['AnswersWithRag'] = query_engineResponse
        return df

print("start measuring time")
start = time.time()

new_df=modelResponse(llm,query_engine,df)
print("end of LLM answers")
end = time.time()
print(f"Time for inference: {end - start}")
new_df.to_csv("Mistral_Experiment2.csv")
