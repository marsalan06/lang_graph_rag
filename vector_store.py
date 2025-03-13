import logging
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from config import Config
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VectorStore:
    def __init__(self):
        self.index_name = Config.PINECONE_INDEX
        self.embedding_model = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY, model="text-embedding-3-small")
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(host=Config.PINECONE_INDEX_HOST)
        logging.info(f"Connected to Pinecone index: {self.index_name}")

    def load_and_index_pdf(self, file_path, namespace="default", source="", chunk_size=500, chunk_overlap=50):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logging.info(f"Loaded PDF from {file_path} with {len(documents)} pages.")

            tenant_id = st.session_state.user.get("tenant_id") if "user" in st.session_state else None

            for doc in documents:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata.update({
                    "source": source or "Unknown",
                    "tenant_id": tenant_id  # None if not present
                })

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(documents)
            logging.info(f"Split into {len(docs)} chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}.")

            PineconeVector.from_documents(
                docs,
                index_name=self.index_name,
                embedding=self.embedding_model,
                namespace=namespace
            )
            logging.info(f"Indexed {len(docs)} documents into Pinecone namespace '{namespace}' with source '{source}' and tenant_id '{tenant_id}'.")

        except Exception as e:
            logging.error(f"Error loading and indexing PDF: {e}")
            raise

    def index_documents(self, documents, namespace="default", source_filter=""):
        try:
            tenant_id = st.session_state.user.get("tenant_id") if "user" in st.session_state else None

            vectors = []
            for i, doc in enumerate(documents):
                embedding = self.embedding_model.embed_query(doc)
                vectors.append(
                    {
                        "id": f"doc-{i}",
                        "values": embedding,
                        "metadata": {
                            "source": source_filter or "Unknown",
                            "tenant_id": tenant_id,  # None if not present
                            "text": doc
                        },
                        "namespace": namespace
                    }
                )

            self.index.upsert(vectors)
            logging.info(f"Documents indexed successfully in namespace '{namespace}' with tenant_id '{tenant_id}'.")
        except Exception as e:
            logging.error(f"Error indexing documents: {e}")
            raise

    def describe_index_stats(self):
        try:
            stats = self.index.describe_index_stats()
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
            return stats
        except Exception as e:
            logging.error(f"Error retrieving index stats: {e}")
            return None