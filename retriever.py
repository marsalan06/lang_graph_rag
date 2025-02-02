import logging
from pinecone import Pinecone
from config import Config
from langchain_openai import OpenAIEmbeddings

class Retriever:
    """
    Retrieves relevant documents from Pinecone using similarity search.
    """
    def __init__(self):
        self.index_name = Config.PINECONE_INDEX
        self.embedding_model = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY, model="text-embedding-3-small")

        # Initialize Pinecone GRPC client
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Set Pinecone Index Host
        self.index_host = Config.PINECONE_INDEX_HOST
        
        # Connect to the index
        self.index = self.pc.Index(host=self.index_host)

    def retrieve_relevant_docs(self, query, namespace="default", metadata_filter=None, k=3):
        """
        Retrieves the top-k most relevant documents from Pinecone for a given query,
        filtered by namespace and metadata.
        """
        try:
            query_embedding = self.embedding_model.embed_query(query)

            response = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                filter=metadata_filter,
                top_k=k,
                include_metadata=True
            )

            formatted_results = []
            for match in response.get("matches", []):
                doc_text = match.get("metadata", {}).get("text", "No content available")
                source = match.get("metadata", {}).get("source", "Unknown Source")
                formatted_results.append(f"🔹 **Source:** {source}\n📄 **Extract:** {doc_text}\n")

            return "\n\n".join(formatted_results)
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            raise