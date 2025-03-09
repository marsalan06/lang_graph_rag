import logging
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VectorStore:
    """
    Handles Pinecone initialization, document indexing, and connection management.
    """

    def __init__(self):
        self.index_name = Config.PINECONE_INDEX
        self.embedding_model = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY, model="text-embedding-3-small")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_INDEX, embeddings=self.embedding_model)
        self.index = self.pc.Index(host=Config.PINECONE_INDEX_HOST)
        logging.info(f"Connected to Pinecone index: {self.index_name}")

    def load_and_index_pdf(self, file_path, namespace="default", source="", chunk_size=500, chunk_overlap=50):
        """
        Loads a PDF, splits it into chunks, adds metadata, and indexes it into Pinecone.

        Args:
            file_path (str): Path to the uploaded PDF file.
            namespace (str): Pinecone namespace to store the documents.
            source (str): Metadata source value to tag the documents.
            chunk_size (int): Size of each document chunk.
            chunk_overlap (int): Overlap between chunks.
        """
        try:
            # Load the PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logging.info(f"Loaded PDF from {file_path} with {len(documents)} pages.")

            # Add metadata to each document
            for doc in documents:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["source"] = source or "Unknown"

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(documents)
            logging.info(f"Split into {len(docs)} chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}.")

            # Index into Pinecone
            PineconeVector.from_documents(
                docs,
                index_name=self.index_name,
                embedding=self.embedding_model,
                namespace=namespace
            )
            logging.info(f"Indexed {len(docs)} documents into Pinecone namespace '{namespace}' with source '{source}'.")

        except Exception as e:
            logging.error(f"Error loading and indexing PDF: {e}")
            raise

    def index_documents(self, documents, namespace="default",source_filter=""):
        """
        Embeds and indexes documents in Pinecone.
        """
        try:
            vectors = []
            for i, doc in enumerate(documents):
                embedding = self.embedding_model.embed_query(doc)
                vectors.append(
                    {
                        "id": f"doc-{i}",
                        "values": embedding,
                        "metadata": {"source": source_filter},
                        "namespace": namespace
                    }
                )

            self.index.upsert(vectors)
            logging.info(f"Documents indexed successfully in namespace '{namespace}'.")
        except Exception as e:
            logging.error(f"Error indexing documents: {e}")
            raise

    def describe_index_stats(self):
        """
        Retrieves and prints statistics of the Pinecone index.
        """
        try:
            stats = self.index.describe_index_stats()

            # Extract key information
            total_vectors = stats.get("total_vector_count", 0)
            namespaces = stats.get("namespaces", {})
            dimensions = stats.get("dimension", "Unknown")
            memory_usage = stats.get("memory_usage", "Unknown")

            formatted_stats = (
                f"\n📊 **Pinecone Index Stats:**\n"
                f"🔹 **Index Name:** {self.index_name}\n"
                f"📌 **Total Vectors:** {total_vectors}\n"
                f"📏 **Vector Dimensions:** {dimensions}\n"
                f"🗂️ **Namespaces & Counts:** {namespaces}\n"
                f"💾 **Memory Usage:** {memory_usage}\n"
            )
            
            logging.info(formatted_stats)
            return stats  # Return the raw stats dictionary, not the formatted string
        except Exception as e:
            logging.error(f"Error retrieving index stats: {e}")
            return None