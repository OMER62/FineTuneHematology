from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import NodeParser, SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from llama_index.core import download_loader, service_context, VectorStoreIndex, StorageContext
from llama_index.legacy.vector_stores import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Check if the index exists, if not, create one
if 'hematology-index' not in pc.list_indexes().names():
    pc.create_index(
        name='my_index',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

if __name__ == '__main__':
    print("Going to ingest pinecone documentation...")
    print(f"{pc.list_indexes().names()}")
    PDFReader = download_loader("PDFReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./Hematology-pdf", file_extractor={".pdf": PDFReader()}
    )
    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=200)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)

    pinecone_index = pc.Index(name="hematology-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("finished ingesting...")
