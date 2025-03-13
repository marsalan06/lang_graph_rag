import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
    LMS_AUTH_URL = os.getenv("LMS_AUTH_URL")



    @staticmethod
    def validate():
        if not Config.PINECONE_API_KEY or not Config.OPENAI_API_KEY:
            raise ValueError("Missing API Keys! Please check your .env file.")
    @staticmethod
    def initialize_firebase():
        if not firebase_admin._apps:
            cred = credentials.Certificate(Config.FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    
# Validate API keys on import
Config.validate()
db = Config.initialize_firebase()

