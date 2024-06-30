import dotenv
import os
import pandas as pd
import time
import torch
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

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



SYSTEM_PROMPT = """You are a medical doctor that specializes in Hematology. 
Return only the letter corresponding to the correct answer to the following question,
do not reply using a complete sentence and only give the answer in the following format: '(x)':
"""
query_wrapper_prompt = PromptTemplate(
    "<s>[INST] <<SYS>>\n" + SYSTEM_PROMPT + "\n<</SYS>>\n\n{query_str} [/INST]"
)
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},)

summarizer = TreeSummarize(llm=llm)

# p = QueryPipeline(verbose=True)


# p.add_modules(
#     {
#         "input": query_wrapper_prompt,
#         "retriever": retriever,
#         "summarizer": summarizer,
#     }
# )

# p.add_link("input", "retriever")
# p.add_link("input", "summarizer", dest_key="query_str")
# p.add_link("retriever", "summarizer", dest_key="nodes")
def modelResponse(pinecone_indices, embed_models):

        response_synthesizer = get_response_synthesizer(llm=llm)
        for idx, embed_model_id in enumerate(embed_models):
            df = pd.read_excel("Harrisons Hematology and Oncology 2ed_questionsPage743.xlsx")
            # llm_answers = pd.DataFrame()
            llmResponse_time=[]
            llmResponse = []
            query_engineResponse=[]
            query_engineResponse_time=[]
            index_name = pinecone_indices[idx]
            pinecone_index = pc.Index(name=index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            embed_model = HuggingFaceEmbedding(model_name=embed_model_id)

            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
            # configure retriever
            retriever = VectorIndexRetriever(index=index, similarity_top_k=10,)


            query_engine = RetrieverQueryEngine(retriever=retriever,
                                                response_synthesizer=response_synthesizer,
                                node_postprocessors=[SentenceEmbeddingOptimizer(
                                      embed_model=embed_model,
                                      percentile_cutoff=0.7)],)
            questions=df['Questions']
            for question in questions:
                start = time.time()
                answer = llm.complete(question).text
                end = time.time()
                llmResponse_time.append(end - start)
                llmResponse.append(answer)
                print("answer")
                #start = time.time()
                #query_answer = query_engine.query(question).response
                #end = time.time()
                #query_engineResponse_time.append(end - start)
                #query_engineResponse.append(query_answer)
                print("queryEngineAnswer")
            df['llama2_Base'] = llmResponse
            #df['llama_Rag'] = query_engineResponse
            df['timeBase'] = llmResponse_time
            #df['timeRag'] = query_engineResponse_time
            df.to_csv(f"llama2-7b_{pinecone_indices[idx]}_BaseAnswersOnly.csv")

        
#LLM_QUERY = """A 39-year-old woman is evaluated for anemia. Her  laboratory studies reveal a hemoglobin of 7.4 g/dL,  hematocrit of 23.9%, mean corpuscular volume of  72 fL, mean cell hemoglobin of 25 pg, and mean cell  hemoglobin concentration of 28%. The peripheral  smear is shown in Figure 2. Which of the follow- ing tests is most likely to be abnormal in this patient?  A. Ferritin B. Haptoglobin C. Hemoglobin electrophoresis D. Glucose-6-phosphate dehydrogenase E. Vitamin B12"""
#LLM_QUERY = "How is the weather up there?"
print("start measuring time")
start = time.time()
modelResponse(['hematology-index-llama-high'], 
              [embed_model_id_c])
print("end of LLM answers")
end = time.time()
print(f"Time for inference: {end - start}")
