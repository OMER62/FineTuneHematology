import json

from dotenv import load_dotenv
import os

from llama_index.core.node_parser import JSONNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.service_context import ServiceContext
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.legacy.vector_stores import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Check if the index exists, if not, create one
if 'medical-cases-index' not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name='medical-cases-index',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

if __name__ == '__main__':
    print("Going to ingest data into pinecone index...")
    print(f"Available indexes: {pc.list_indexes().names()}")

    # Load JSON data from a file
    input_file = './train.json'

    documents = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Convert JSON string to dictionary
                data = json.loads(line)
                data.pop('id', None)
                data.pop('sent2', None)

                # Rename the 'sent1' key to 'question'
                data['Question'] = data.pop('sent1')
                data['Option0'] = data.pop('ending0')
                data['Option1'] = data.pop('ending1')
                data['Option2'] = data.pop('ending2')
                data['Option3'] = data.pop('ending3')
                data['Answer'] = "option"+str(data.pop('label'))

                # Convert the dictionary back to a JSON string
                json_str_renamed = json.dumps(data, indent=4)

                documents.append(Document(text=json_str_renamed))
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON on line: {line}")
                print(f"Error: {e}")

    # Initialize parser and other components
    parser = JSONNodeParser(documents=documents)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=parser)

    # Connect to Pinecone index
    pinecone_index = pc.Index(name="medical-cases-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # nodes = parser.get_nodes_from_documents(documents)
    # Process documents and index them
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    print("Finished ingesting...")