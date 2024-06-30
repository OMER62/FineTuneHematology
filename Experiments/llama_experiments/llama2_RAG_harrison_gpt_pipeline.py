from dotenv import load_dotenv
import os
import pandas as pd
import time
import torch
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceEmbeddingOptimizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.postprocessor.cohere_rerank import CohereRerank

load_dotenv()
# let's try avoiding fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
login(token=os.environ.get("LOGIN_TOKEN"))

pc = Pinecone(api_key=os.environ.get("PINECONE_BUSAKUZA_API"))

#embed_model_id_a = 'sentence-transformers/all-MiniLM-L6-v2'
#embed_model_id_b = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
#embed_model_id = 'sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja'
embed_model_id = "text-embedding-ada-002"


from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"


selected_model = LLAMA2_7B_CHAT



# SYSTEM_PROMPT = """You are a medical doctor that specializes in Hematology. 
# Return only the letter corresponding to the correct answer to the following question, 
# do not reply using a complete sentence and only give the answer in the following format: '(x)':
# """
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

# Get Vectore index from pinecone and configure the retreiver
pinecone_index = pc.Index(name='hematology-index-llama-harrison-gpt')
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
embed_model = OpenAIEmbedding(model=embed_model_id, api_key=os.environ.get("OPENAI_API"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
retriever = VectorIndexRetriever(index=index, similarity_top_k=5,)
response_synthesizer = get_response_synthesizer(llm=llm)
query_engine = RetrieverQueryEngine(retriever=retriever,
                                    response_synthesizer=response_synthesizer,
                                    node_postprocessors=[SentenceEmbeddingOptimizer(
                                    embed_model=embed_model, percentile_cutoff=0.7)],)


#reranker = CohereRerank()

# Configure The Query Pipeline for Rag
p = QueryPipeline(verbose=True)
summarizer = TreeSummarize(llm=llm)
p.add_modules({
                "input": query_wrapper_prompt_pipeline,
                "retriever": retriever,
                "summarizer": summarizer,})

p.add_link("input", "retriever")
p.add_link("input", "summarizer", dest_key="query_str")
p.add_link("retriever", "summarizer", dest_key="nodes")



def modelResponse(engine, query_pipeline, llm):

        df = pd.read_csv("llama2-7b_hematology-index-llama-high_BaseAnswersOnly.csv")
        # llm_answers = pd.DataFrame()
        llm_responses = []
        llm_responses_time = []
        pipeline_responses=[]
        pipeline_response_time=[]
        query_engineResponses=[]
        query_engineResponse_time=[]
        questions=df['Questions']
        for q_id, question in enumerate(questions):
            # pipeline
            start = time.time()
            pipeline_answer = query_pipeline.run(topic=question)
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
            query_engine_answer = engine.query(question).response 
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
        df.to_csv(f"llama2-7b_hematology-index-llama-harrison-gpt-piplineAndRag_student.csv")

        

print("start measuring time")
start = time.time()
modelResponse(query_engine, p, llm)
print("end of LLM answers")
end = time.time()
print(f"Time for inference: {end - start}")
