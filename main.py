import io
import requests
import fitz  # PyMuPDF
from pathlib import Path
from scrapper import scrape_google_scholar_pdfs
from populate_database import main as populate_db
from langchain.schema.document import Document

def extract_text_from_url(url):
    """Extract text directly from a PDF URL without downloading"""
    try:
        # Get PDF content from URL
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch PDF from {url}")
            return ""

        # Create a memory buffer from the response content
        pdf_stream = io.BytesIO(response.content)
        
        # Open PDF from memory buffer
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        doc.close()
        return text
        
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return ""

def prepare_data(query_text: str):
    """Process PDFs directly from URLs"""
    # Get PDF links
    pdf_links = scrape_google_scholar_pdfs(query_text, max_results=5)
    
    # Process PDFs directly from URLs
    documents = []
    for i, url in enumerate(pdf_links):
        print(f"Processing PDF from URL: {url}")
        text = extract_text_from_url(url)
        if text:
            # Create a Document object with the extracted text and metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "page": 0,  # Since we're processing the entire PDF as one document
                }
            )
            documents.append(doc)
            print(f"Successfully processed PDF from {url}")
        else:
            print(f"Failed to process PDF from {url}")

    # Process the documents and update the database
    if documents:
        process_documents(documents)
    else:
        print("No documents were successfully processed")

def process_documents(documents):
    """Process the documents and update the database"""
    from populate_database import split_documents, add_to_chroma
    
    # Split the documents into chunks
    chunks = split_documents(documents)
    
    # Add to Chroma database
    add_to_chroma(chunks)

if __name__ == "__main__":
    query = input("Enter your search query: ")
    prepare_data(query)
