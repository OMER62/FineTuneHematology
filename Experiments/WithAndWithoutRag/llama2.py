import dotenv
import os
import pandas as pd
import time
from huggingface_hub import login

# access HuggingFace to get access to the LLMs
# access_token = os.environ.get("LOGIN_TOKEN")
from llama_index.core.llms import LLM
from transformers import BitsAndBytesConfig

login(token="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")

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.service_context import ServiceContext
from llama_index.core.node_parser import NodeParser, SimpleNodeParser
from pinecone import Pinecone
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings


# get the Hematology-index created by gpt-3.5 embeddings
pc = Pinecone(api_key="e907ba23-c248-44b1-8a41-7ac1ab90cfc1")
pinecone_index = pc.Index(name="hematology-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key="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")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# configure a vector retriever
retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

# Model names (make sure you have access on HF)
LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
LLAMA2_13B_CHAT = "/home/kazadan/llama-2-13b-chat"
LLAMA2_70B = "meta-llama/Llama-2-70b-hf"
LLAMA2_70B_CHAT = "meta-llama/Llama-2-70b-chat-hf"

selected_model = LLAMA2_7B_CHAT

# create a system prompt
SYSTEM_PROMPT = """You are a medical AI assistant that specializes in Hematology. Give an answer with explanation to the following question:
"""

query_wrapper_prompt = PromptTemplate(
    "<s>[INST] <<SYS>>\n" + SYSTEM_PROMPT + "\n<</SYS>>\n\n{query_str} [/INST]"
)



# configure LLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    query_wrapper_prompt=query_wrapper_prompt,
    #system_prompt=SYSTEM_PROMPT,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit":True},
)
node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=200)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)


Settings.llm = llm
# start = time.time()
# print("Question:")
# print(llm.complete("What is hematology?").text)
# end = time.time()
# print(f"answer: {end - start}")
response_synthesizer = get_response_synthesizer(llm=llm, service_context=service_context)
df = pd.read_excel("Harrisons Hematology and Oncology 2ed_questionsPage743.xlsx")

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
            print(f"question: {q}")
            answer = llm.complete(q).text
            llmResponse.append(answer)
            print("answer")
            query_answer = query_engine.query(q).response
            query_engineResponse.append(query_answer)
            print("queryEngineAnswer")
        df['llama2-7b'] = llmResponse
        # print(f"The response without RAG:\n{answer}\n")
        # print(f"The response after RAG:\n{query_answer}\n")

        df['AnswersWithRag'] = query_engineResponse
        return df

print("start measuring time")
start = time.time()

new_df=modelResponse(llm,query_engine,df)
print("end of LLM answers")
end = time.time()
print(f"Time for inference: {end - start}")
new_df.to_csv("llama2-7b_Experiment2.csv")