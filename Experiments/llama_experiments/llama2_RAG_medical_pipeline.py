from dotenv import load_dotenv
import os
import pandas as pd
import time
import torch
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceEmbeddingOptimizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()
# let's try avoiding fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
login(token=os.environ.get("LOGIN_TOKEN"))

pc = Pinecone(api_key=os.environ.get("PINECONE_BUSAKUZA_API"))



embed_model_id_a = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_id_b = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
embed_model_id_c = 'sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja'
embed_model_id_d = "text-embedding-ada-002"

import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"


selected_model = LLAMA2_7B_CHAT



#SYSTEM_PROMPT = """You are a medical doctor that specializes in Hematology. Return only the letter corresponding to the correct answer to the following question, do not reply using a complete sentence and only give the answer in the following format: '(x)':
#"""
SYSTEM_PROMPT = """You are a medical student taking an exam in Hematology. 
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

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    tokenizer_name=selected_model,
    query_wrapper_prompt=query_wrapper_prompt_llm,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},)

summarizer = TreeSummarize(llm=llm)

def modelResponse(pinecone_indices, embed_models, llm):

        response_synthesizer = get_response_synthesizer(llm=llm)
        for idx, embed_model_id in enumerate(embed_models):
            df = pd.read_excel("final_questionaire.xlsx").iloc[0:500]
            # lists for answers via base model, rag, pipeline
            llm_responses = []
            llm_responses_time=[]
            pipeline_responses=[]
            pipeline_response_time=[]
            query_engineResponses=[]
            query_engineResponse_time=[]
            
            # configure Index
            index_name = pinecone_indices[idx]
            pinecone_index = pc.Index(name=index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
            # embed_model = OpenAIEmbedding(model=embed_model_id, api_key=os.environ.get("OPENAI_API"))
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
            # configure retriever
            retriever = VectorIndexRetriever(index=index, similarity_top_k=5,)
            
            p = QueryPipeline(verbose=True)


            p.add_modules(
            {
                "input": query_wrapper_prompt_pipeline,
                "retriever": retriever,
                "summarizer": summarizer,
            })

            p.add_link("input", "retriever")
            p.add_link("input", "summarizer", dest_key="query_str")
            p.add_link("retriever", "summarizer", dest_key="nodes")

            query_engine = RetrieverQueryEngine(retriever=retriever,
                                                response_synthesizer=response_synthesizer,
                                node_postprocessors=[SentenceEmbeddingOptimizer(
                                      embed_model=embed_model,
                                      percentile_cutoff=0.7)],)
            
            questions=df['Questions']
            for q_id, question in enumerate(questions):
                # pipeline
                start = time.time()
                pipeline_answer = p.run(topic=question)
                end = time.time()
                pipeline_response_time.append(end - start)
                pipeline_responses.append(pipeline_answer)
                print(f"pipeline answer({q_id + 1}) {end - start}")
                # llm_base
                start = time.time()
                llm_answer = llm.complete(question).text
                end = time.time()
                llm_responses_time.append(end - start)
                llm_responses.append(llm_answer)
                print(f"llm answer({q_id + 1}) {end - start}")
                torch.cuda.empty_cache()
                # rag
                start = time.time()
                query_engine_answer = query_engine.query(question).response 
                end = time.time()
                query_engineResponse_time.append(end - start)
                query_engineResponses.append(query_engine_answer)
                print(f"queryEngineAnswer({q_id + 1}) {end - start}")
            df['llama2_Base'] = llm_responses
            df['llama_Qpipeline'] = pipeline_responses
            df['llama_Rag'] = query_engineResponses
            df['timeBase'] = llm_responses_time
            df['timeQPipe'] = pipeline_response_time
            df['timeRag'] = query_engineResponse_time
            
            df.to_csv(f"llama2-7b_{pinecone_indices[idx]}_piplineAndRag_full.csv", escapechar="\\")

        

print("start measuring time")
start = time.time()
modelResponse(['hematology-index-full'], 
              [embed_model_id_c], llm)
print("end of LLM answers")
end = time.time()
print(f"Time for inference: {end - start}")
