import logging
from pinecone import Pinecone
from config import Config
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document  # Required for LangGraph processing

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

    def retrieve_relevant_docs(self, query, namespace="default", metadata_filter=None, k=3, retry=True):
        """
        Retrieves the top-k most relevant documents from Pinecone for a given query,
        filtered by namespace and metadata.

        Args:
            query (str): The user's query.
            namespace (str): The namespace for Pinecone filtering.
            metadata_filter (dict): Optional filter for metadata.
            k (int): Number of documents to retrieve.
            retry (bool): Whether to retry with increased `k` if no results are found.

        Returns:
            list[Document]: A list of LangChain Document objects.
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

            matches = response.get("matches", [])
            if not matches and retry:
                logging.warning(f"No results found for query: '{query}'. Retrying with k={k+2}")
                return self.retrieve_relevant_docs(query, namespace, metadata_filter, k=k+2, retry=False)

            # Convert to LangChain Document format for LangGraph compatibility
            retrieved_docs = []
            for match in matches:
                doc_text = match.get("metadata", {}).get("text", "No content available")
                source = match.get("metadata", {}).get("source", "Unknown Source")
                doc_score = match.get("score", 0)

                retrieved_docs.append(
                    Document(
                        page_content=doc_text,
                        metadata={"source": source, "score": doc_score}
                    )
                )

            logging.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
            return retrieved_docs

        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []
