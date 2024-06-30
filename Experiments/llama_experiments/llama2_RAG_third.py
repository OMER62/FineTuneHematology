import dotenv
import os
import pandas as pd
import time
import torch
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')
# let's try avoiding fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
login(token="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")

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

pc = Pinecone(api_key="a1cb9e3e-19ae-4943-bb3a-78706e3b0394")



embed_model_id_a = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_id_b = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
embed_model_id_c = 'sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja'


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"


selected_model = LLAMA2_7B_CHAT



SYSTEM_PROMPT = """You are a medical doctor that specializes in Hematology. Return only the letter corresponding to the correct answer to the following question, do not reply using a complete sentence and only give the answer in the following format: '(x)':
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

def modelResponse(pinecone_indices, embed_models):

        response_synthesizer = get_response_synthesizer(llm=llm)
        for idx, embed_model_id in enumerate(embed_models):
            df = pd.read_csv("llama2-7b_hematology-index-llama-high_BaseAnswersOnly.csv")
            # save only the letters of the answer
            df["Answers"] = [answer.split("The answer is ")[1][0] for answer in df["Answers"]]
            pipeline_responses=[]
            pipeline_response_time=[]
            query_engineResponses=[]
            query_engineResponse_time=[]
            index_name = pinecone_indices[idx]
            pinecone_index = pc.Index(name=index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            embed_model = HuggingFaceEmbedding(model_name=embed_model_id)

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
            for question in questions:
                start = time.time()
                pipeline_answer = p.run(topic=question)
                end = time.time()
                pipeline_response_time.append(end - start)
                pipeline_responses.append(pipeline_answer)
                
                print("answer")
                start = time.time()
                torch.cuda.empty_cache()
                query_engine_answer = query_engine.query(question).response 
                end = time.time()
                query_engineResponse_time.append(end - start)
                query_engineResponses.append(query_engine_answer)
                print("queryEngineAnswer")
            df['llama_Qpipeline'] = pipeline_responses
            df['llama_Rag'] = query_engineResponses
            df['timeQPipe'] = pipeline_response_time
            df['timeRag'] = query_engineResponse_time
            df.to_csv(f"llama2-7b_{pinecone_indices[idx]}_piplineAndRag.csv")

        

print("start measuring time")
start = time.time()
modelResponse(['hematology-index-llama-third'], 
              [embed_model_id_c])
print("end of LLM answers")
end = time.time()
print(f"Time for inference: {end - start}")
