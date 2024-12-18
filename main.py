import io
import asyncio
import aiohttp
import fitz  # PyMuPDF
from pathlib import Path
from scrapper import scrape_google_scholar_pdfs
from populate_database import main as populate_db
from langchain.schema.document import Document
import torch
from tqdm import tqdm
from rake_nltk import Rake
import nltk
from itertools import combinations

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

# Initialize NLTK data
download_nltk_data()

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def extract_keywords(description):
    """Extract keywords from company description using RAKE algorithm"""
    try:
        r = Rake(
            min_length=1,  # Minimum phrase length
            max_length=4   # Maximum phrase length
        )
        r.extract_keywords_from_text(description)
        
        # Get the top keywords based on scores
        keywords = r.get_ranked_phrases()[:5]  # Get top 5 keywords/phrases
        
        # Generate search queries from keywords
        search_queries = []
        
        # Add individual keywords
        search_queries.extend(keywords)
        
        # Add combinations of two keywords
        keyword_pairs = list(combinations(keywords, 2))
        for pair in keyword_pairs[:3]:  # Limit to top 3 combinations
            search_queries.append(" ".join(pair))
        
        return search_queries
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        # Return a simple fallback if keyword extraction fails
        return [description]

async def extract_text_from_url(session, url, timeout=30):
    """Extract text directly from a PDF URL without downloading"""
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                print(f"Failed to fetch PDF from {url}")
                return None, url

            content = await response.read()
            pdf_stream = io.BytesIO(content)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text, url
            
    except asyncio.TimeoutError:
        print(f"Timeout while processing {url}")
        return None, url
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None, url

async def process_urls_concurrently(urls):
    """Process multiple URLs concurrently"""
    documents = []
    async with aiohttp.ClientSession() as session:
        tasks = [extract_text_from_url(session, url) for url in urls]
        
        for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing PDFs"):
            text, url = await completed_task
            if text:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "page": 0
                    }
                )
                documents.append(doc)
                print(f"Successfully processed PDF from {url}")
            else:
                print(f"Failed to process PDF from {url}")
    
    return documents

def prepare_data(company_description: str):
    """Process PDFs based on company description keywords"""
    # First, create a Document from the company description
    company_doc = Document(
        page_content=company_description,
        metadata={
            "source": "company_description",
            "page": 0
        }
    )
    
    # Extract keywords and generate search queries
    search_queries = extract_keywords(company_description)
    print("\nGenerated search queries:", search_queries)
    
    all_pdf_links = set()  # Use set to avoid duplicates
    
    # Search for PDFs using each query
    for query in search_queries:
        print(f"\nSearching for: {query}")
        pdf_links = scrape_google_scholar_pdfs(query, max_results=5)
        all_pdf_links.update(pdf_links)
    
    # Convert set back to list and limit total PDFs
    pdf_links = list(all_pdf_links)[:5]
    
    # Process PDFs concurrently
    documents = asyncio.run(process_urls_concurrently(pdf_links))
    
    # Add company description to the beginning of documents list
    documents.insert(0, company_doc)

    # Process all documents including company description
    if documents:
        process_documents(documents)
    else:
        print("No documents were successfully processed")

def process_documents(documents):
    """Process the documents and update the database"""
    from populate_database import split_documents, add_to_chroma
    
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    print("Adding chunks to database...")
    add_to_chroma(chunks)

if __name__ == "__main__":
    description = input("Please provide a description of what your company does: ")
    prepare_data(description)
