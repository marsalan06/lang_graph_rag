import logging
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import Config

class Retriever:
    """
    Retrieves relevant documents from Pinecone using similarity search.
    """

    def __init__(self):
        self.index_name = Config.PINECONE_INDEX
        self.embedding_model = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY, model="text-embedding-3-small")
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(host=Config.PINECONE_INDEX_HOST)

    def retrieve_relevant_docs(self, query, namespace="default", metadata_filter=None, k=3) -> list:
        try:
            query_embedding = self.embedding_model.embed_query(query)
            print("-------------query_embedding-------")
            print(query_embedding)
            print("-------------metadata_filter-------")
            print(metadata_filter)
            print("-------------k-------")
            print(k)
            print("-------------namespace-------")
            print(namespace)
            response = self.index.query(namespace=namespace, vector=query_embedding, filter=metadata_filter, top_k=k, include_metadata=True)

            retrieved_docs = [
                Document(page_content=match["metadata"].get("text", "No content available"), metadata={"source": match["metadata"].get("source", "Unknown")})
                for match in response.get("matches", [])
            ]
            print("-------------retrived docs-------")
            print(retrieved_docs)
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []
