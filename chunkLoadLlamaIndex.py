"""
- The simple directory reader:
    The simple directory reader class is a crucial data connector within the llama index,
    designed to facilitate the loading of documents from directories. Its key features include the ability
    to recursively iterate through directories and to filter files by specific extensions. This functionality
    is particularly useful for loading the llama index documentation.
- NodeParser:
    This class underpins the llama index by segmenting documents into 'index nodes',
    smaller parts enriched with metadata about their source document and node sequence.
    It also features node parsers for customizable text splitting based on file type,
    ensuring tailored content processing.
- service_context:
    The Serving Context class is crucial in the llama index,
    packaging essential resources for both indexing and querying phases.
    It encompasses the language model/embeddings model information,
    the node parser for data segmentation control, and integrates the Vector Store Index class for
    managing interactions with vector stores, streamlining the indexing pipeline and application
    querying process.
"""
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import download_loader
def load_and_parse_documents_to_nodes(input_dir):
    """
    Loads documents from the specified input directory and parses them into nodes.

    :param input_dir: The path to the directory containing PDF documents.
    :return: A list of nodes parsed from the documents.
    """
    # Assuming download_loader, SimpleDirectoryReader, and SimpleNodeParser
    # are defined elsewhere and imported correctly.

    # Replace 'download_loader' with the actual method to obtain the PDFReader
    PDFReader = download_loader("PDFReader")

    # Initialize the directory reader with the input directory
    dir_reader = SimpleDirectoryReader(
        input_dir=input_dir,  # Use the function's input parameter
        file_extractor={".pdf": PDFReader()}
    )

    # Load documents from the directory
    documents = dir_reader.load_data()

    # Initialize the node parser with default settings and custom chunk size/overlap
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=200)

    # Parse the loaded documents into nodes
    nodes = node_parser.get_nodes_from_documents(documents=documents)

    return nodes