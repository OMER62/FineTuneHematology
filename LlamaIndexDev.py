from dotenv import load_dotenv
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.service_context import ServiceContext
from pinecone import Pinecone
from llama_index.legacy.vector_stores import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext


def initialize_service_context(llm, node_parser, model="text-embedding-ada-002"):
    """
    Initializes the embedding model and service context for LLM-powered applications.

    :param llm: The language model to use within the service context.
    :param node_parser: The node parser to use within the service context.
    :param model: The model identifier for the embedding model. Defaults to "text-embedding-ada-002".
    :return: An instance of ServiceContext configured with the specified LLM, embedding model, and node parser.
    """
    # Initialize the embedding model with the specified model identifier
    embed_model = OpenAIEmbedding(model=model)

    # Create the service context with the provided LLM, embedding model, and node parser
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)

    return service_context


def create_and_ingest_vector_store(documents, service_context, index_name="llamaindex-documentation-helper",
                                   show_progress=True):
    """
    Creates a vector store for the provided documents and ingests them into the Pinecone index.

    :param documents: The documents to be indexed.
    :param service_context: The service context, including the LLM and embedding model.
    :param index_name: The name of the Pinecone index to be created or used. Defaults to "llamaindex-documentation-helper".
    :param show_progress: Flag to show progress during the ingestion process. Defaults to True.
    """
    # Assuming pinecone and PineconeVectorStore are correctly imported and initialized elsewhere
    load_dotenv()
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index(name="hematology-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=show_progress,
    )
    print("Finished ingesting...")
