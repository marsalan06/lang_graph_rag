from vector_store import VectorStore
from retriever import Retriever
from data_loader import DataLoader

def main():
    # Initialize components
    vector_store = VectorStore()
    retriever = Retriever()
    loader = DataLoader()

    # # Load and index a PDF (stored in "research" namespace)
    # pdf_text = loader.load_pdf("sample.pdf")  # Ensure this file exists
    # vector_store.index_documents([pdf_text], namespace="research")

    # # Load and index a Webpage (stored in "web-articles" namespace)
    # web_text = loader.load_webpage("https://en.wikipedia.org/wiki/Machine_learning")
    # vector_store.index_documents([web_text], namespace="web-articles")

    # describe_index_stats
    vector_store.describe_index_stats()

    # Query retrieval with metadata filtering
    query = "What is coupling?"
    metadata_filter = {"source": {"$eq": "software_design_dev"}}
    
    retrieved_docs = retriever.retrieve_relevant_docs(query, namespace="SE_Software_Engineering", metadata_filter=metadata_filter)

    print("\n🔹 **Retrieved Documents:**\n")
    print(retrieved_docs)

if __name__ == "__main__":
    main()
