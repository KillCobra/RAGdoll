import io
import asyncio
import aiohttp
import fitz  # PyMuPDF
from pathlib import Path
from scrapper import scrape_google_scholar_pdfs
from populate_database import main as populate_db
from populate_database import split_documents, add_to_chroma
from langchain.schema.document import Document
import torch
from tqdm import tqdm
from rake_nltk import Rake
import nltk
from itertools import combinations
from query_data import run_query_with_description, is_keyword_used, add_keyword  # Import functions
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import argparse

from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

# Download required NLTK data
def download_nltk_data():
    """Ensure required NLTK datasets are downloaded."""
    nltk_resources = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']

    for resource in nltk_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)

# Initialize NLTK data
# download_nltk_data()

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize Chroma DB for keyword management
CHROMA_PATH = "chroma"
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def extract_keywords(description, max_keywords):
    """Extract keywords from company description using RAKE algorithm"""
    try:
        r = Rake(
            min_length=1,  # Minimum phrase length
            max_length=4   # Maximum phrase length
        )
        r.extract_keywords_from_text(description)
        
        # Get the top keywords based on scores
        keywords = r.get_ranked_phrases()[:max_keywords]  # Get top max_keywords keywords/phrases
        
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

def prepare_data(company_description: str, market_or_sector: str, max_pdf_links: int, max_keywords: int):
    """Process PDFs based on company description keywords and market or sector"""
    # Create a Document from the company description
    company_doc = Document(
        page_content=company_description,
        metadata={
            "source": "company_description",
            "page": 0
        }
    )
    
    # Extract keywords and generate search queries
    search_queries = extract_keywords(company_description, max_keywords)
    
    # Include market or sector as a keyword
    search_queries.append(market_or_sector)
    print("\nGenerated search queries:", search_queries)
    
    all_pdf_links = set()  # Use set to avoid duplicates
    
    # Search for PDFs using each query
    for query in search_queries:
        # Check if the keyword has already been used (exists in Chroma DB)
        if is_keyword_used(db, query):
            print(f"Keyword '{query}' has already been searched. Skipping.")
            continue  # Skip if the keyword has already been searched
        else:
            print(f"\nSearching for: {query}")
            pdf_links = scrape_google_scholar_pdfs(query, max_results=max_pdf_links)
            all_pdf_links.update(pdf_links)
        
        # Mark the keyword as used by adding it to Chroma DB
        add_keyword(db, query)
    
    # Convert set back to list and limit total PDFs
    pdf_links = list(all_pdf_links)[:max_pdf_links]
    
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
    
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    print("Adding chunks to database...")
    add_to_chroma(chunks)

def save_response_to_pdf(response_text, file_path):
    """Save the provided text to a PDF file."""
    c = canvas.Canvas(file_path, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    lines = response_text.split('\n')
    
    for line in lines:
        text.textLine(line)
    
    c.drawText(text)
    c.showPage()
    c.save()
    print(f"Response saved to PDF at {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Process company data and generate reports.")
    parser.add_argument("description", type=str, help="Company description")
    parser.add_argument("market_or_sector", type=str, help="Market or sector for expansion")
    parser.add_argument("--max_keywords", type=int, default=6, help="Maximum number of keywords to extract")
    parser.add_argument("--max_pdf_links", type=int, default=20, help="Maximum number of PDF links to retrieve")
    args = parser.parse_args()

    # Use the arguments in the functions
    prepare_data(
        args.description,
        args.market_or_sector,
        args.max_pdf_links,
        args.max_keywords
    )

    response = run_query_with_description(args.description, args.market_or_sector, query_text=None)
    if response:
        print("\nResponse:", response["text"])
        pdf_path = os.path.join("data", "response.pdf")
        os.makedirs("data", exist_ok=True)
        save_response_to_pdf(response["text"], pdf_path)
    else:
        print("No response generated or an error occurred.")

if __name__ == "__main__":
    main()
