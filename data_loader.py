import os
import logging
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataLoader:
    """
    Handles extraction of text from PDFs and web pages.
    """

    def load_pdf(self, pdf_path):
        """
        Extracts text from a PDF file.
        """
        if not os.path.exists(pdf_path):
            logging.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"File not found: {pdf_path}")

        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            extracted_text = " ".join([page.page_content for page in pages])
            logging.info(f"Extracted {len(extracted_text)} characters from PDF.")
            return extracted_text
        except Exception as e:
            logging.error(f"Error loading PDF: {e}")
            raise

    def load_webpage(self, url):
        """
        Extracts text from a webpage.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract main text content
            paragraphs = soup.find_all("p")
            extracted_text = " ".join([p.get_text() for p in paragraphs])
            
            logging.info(f"Extracted {len(extracted_text)} characters from webpage.")
            return extracted_text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching webpage: {e}")
            raise
