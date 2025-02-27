from pipeline import CRAGPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    crag = CRAGPipeline()
    
    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit'): ")
        if user_query.lower() == "exit":
            break

        logging.info(f"🎤 User Input: {user_query}")
        response = crag.run(user_query)

        print("\n🔹 **Response:**\n")
        print(response)  # ✅ Always prints final response

if __name__ == "__main__":
    main()
