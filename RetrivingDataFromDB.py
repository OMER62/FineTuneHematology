from dotenv import load_dotenv
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
# Initialize Pinecone
if __name__ == '__main__':
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    pinecone_index = pc.Index(name="hematology-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(llm=llm)

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    # query
    response = query_engine.query("What is this hematology?")
    print(response)
